# main.py
# AdaOWQ v3 Execution Script for Sensors Journal Baseline
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm

from adaowq_utils import AdaOWQProfiler
from owq.recon_soark_ena import GPTQ_OWQ
from owq.quant import *
from owq.utils.misc import *
from owq.utils.datautils import *
from owq.utils.modelutils import *


@torch.no_grad()
def layerwise_quantize(model, dataloader, dev, args):
    meta = args.meta
    print('Starting AdaOWQ v3 (Adaptive Mixed-Precision OWQ) ...')

    use_cache = model.config.use_cache
    layers, pre_layers, _ = parsing_layers(model, meta)
    model.config.use_cache = False

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {kw: None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module.cpu()
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache

    owq_layers = args.meta['owq_layers']
    ratios = args.meta['ratios']

    # ==================== Step 1 ====================
    base_n_out_dict = {}
    if args.target_bit is not None:
        n_owq_layers = sum(owq_layers.values())
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits) / n_owq_layers
        layer_base = find_layers(layers[0])
        for l in owq_layers:
            n_out_base = round(layer_base[l].weight.data.shape[1] * r * ratios[l])
            if n_out_base % 2 == 1:
                n_out_base += 1
            base_n_out_dict[l] = n_out_base
    elif args.target_rank is not None:
        for l in owq_layers:
            n_out_base = args.target_rank
            if n_out_base % 2 == 1:
                n_out_base += 1
            base_n_out_dict[l] = n_out_base
    else:
        for l in owq_layers:
            base_n_out_dict[l] = 0

    # ==================== Step 2====================
    print("Stage 1: Hessian-Activation Sensitivity Profiling...")
    profiler = AdaOWQProfiler(alpha=0.6, beta=0.4, guarantee_ratio=0.5)
    layer_sensitivities = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer)

        gptq_owq_temp = {}
        for name in owq_layers:
            if name in block_layers:
                gptq_owq_temp[name] = GPTQ_OWQ(
                    block_layers[name], n_out=0,
                    expansion_factor=1.0
                )

        def add_batch_temp(name):
            def tmp(_, inp, out):
                gptq_owq_temp[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [block_layers[name].register_forward_hook(add_batch_temp(name))
                   for name in gptq_owq_temp]
        for j in range(min(args.nsamples, 16)):
            layer(inps[j].unsqueeze(0), **inp_kwargs)
        for h in handles:
            h.remove()

        for name in gptq_owq_temp:
            full_name = f"{i}.{name}"
            sens = profiler.compute_layer_sensitivity(
                gptq_owq_temp[name], block_layers[name], inps[:16]
            )
            layer_sensitivities[full_name] = sens
            gptq_owq_temp[name].free()

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    # ==================== Step 3 ====================
    print("Stage 2: Adaptive Budget Allocation (Zero Global Extra Budget)...")
    n_out_dict = profiler.allocate_budgets(
        layer_sensitivities, base_n_out_dict, temp=5.0
    )

    # ==================== Step 4====================
    print("Stage 3: Quantizing with AdaOWQ Engine...")
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer)

        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(block_layers.keys())]

        for names in sequential:
            subset = {n: block_layers[n] for n in names}
            gptq_owq = {}

            for name in subset:
               
                current_n_out = n_out_dict.get(f"{i}.{name}", base_n_out_dict.get(name, 0))
                gptq_owq[name] = GPTQ_OWQ(
                    subset[name], n_out=current_n_out,
                    expansion_factor=args.expansion_factor,
                    tier1_ratio=args.tier1_ratio,
                    tier2_bits=args.tier2_bits,
                    sym=args.sym
                )
                gptq_owq[name].quantizer = Quantizer(
                    args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                )
                gptq_owq[name].quantizer.n_out = current_n_out

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_owq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), **inp_kwargs)
            for h in handles:
                h.remove()

            # weak column
            for name in subset:
                if not args.no_frob_norm:
                    W = subset[name].weight.data.clone().to(torch.float)
                    temp_quantizer = Quantizer(
                        args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                    )
                    temp_quantizer.find_params(W, weight=True, num=40)
                    W_quant = temp_quantizer.quantize(W)
                    frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                    del W, W_quant, temp_quantizer
                else:
                    frob_norm_error = None

                out_ids = gptq_owq[name].hessian_sorting(
                    actorder=args.act_order,
                    frob_norm=frob_norm_error,
                    tier1_ratio=args.tier1_ratio
                )
                gptq_owq[name].quantizer.out_ids = out_ids

     
            for name in subset:
                n_allocated = n_out_dict.get(f'{i}.{name}', 0)
                print(f"Quantizing {meta['prefix']}.{i}.{name} (Allocated Outlier Cols={n_allocated})")
                gptq_owq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize,
                    actorder=args.act_order
                )
                quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                gptq_owq[name].free()

            if not args.no_frob_norm:
                torch.cuda.empty_cache()

        for name in list(block_layers.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

        layers[i] = layer.cpu()
        del layer, gptq_owq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


@torch.no_grad()
def eval_ppl(model, testenc, dev, args):
    meta = args.meta
    print('Evaluating PPL...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // args.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers, pre_layers, post_layers = parsing_layers(model, meta)

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {kw: None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * args.seqlen):((i + 1) * args.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    for post_layer in post_layers:
        post_layer = post_layer.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        for post_layer in post_layers:
            hidden_states = post_layer(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * args.seqlen):((i + 1) * args.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
    print(f"PPL: {ppl.item():.4f}")

    model.config.use_cache = use_cache
    return ppl.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='hugging face model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'custom_path'])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--wbits', type=int, default=3, choices=[2, 3, 4, 16])
    parser.add_argument('--target_bit', type=float, default=3.01)
    parser.add_argument('--target_rank', type=int, default=None)

   
    parser.add_argument('--expansion_factor', type=float, default=1.0, help='Set to 1.0 for standard OWQ export format')
    parser.add_argument('--tier1_ratio', type=float, default=0.5)
    parser.add_argument('--tier2_bits', type=int, default=8)

   
    parser.add_argument('--fake', action='store_true', help='Save fake quantized checkpoint.')
    parser.add_argument('--packing', action='store_true', help='Whether to save 3bit quantized model.')
    parser.add_argument('--faster', action='store_true', help='Whether to save and load 3bit quantized model using faster kernel.')
    parser.add_argument('--logfile', type=str, default='', help='Logging file name')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--layers', nargs='+', type=str, default=None, help='Layers to apply OWQ.')

    parser.add_argument('--tuning', type=str, default='mse', choices=['mse', 'minmax'])
    parser.add_argument('--no_frob_norm', action='store_true')
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--nearest', action='store_true')
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--true-sequential', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')

    args = parser.parse_args()
    meta = processing_arguments(args)
    args.meta = meta
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_all(args.seed)

    if args.load:
        model = load_model(args.model, args.load, False)
    else:
        model = get_hfmodel(args.model, args.dtype)

    if getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = model.config.max_position_embeddings
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = model.config.max_sequence_length
    else:
        args.seqlen = 2048

    if not args.load and args.wbits < 16 and not args.nearest:
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True
        )
        tick = time.time()
        quantizers = layerwise_quantize(model, dataloader, device, args)
        print(f"Running Time : {round((time.time() - tick), 1)}s")

    if not args.no_eval:
        ppl_tasks = ['wikitext2', 'ptb', 'c4']
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, train=False
            )
            print(f"Evaluated Dataset: {dataset}")
            eval_ppl(model, testloader, device, args)

    if args.save:
        save_model(model, quantizers, args.save, False, True)


        ###########
