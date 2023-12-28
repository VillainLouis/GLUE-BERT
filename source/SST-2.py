import torch
import transformers
from SST.SST import SST
from SST.SST_reader import SST_reader
from absl import app, flags
import os
import math
from tqdm import tqdm

from numpy import mean
import loralib as lora

import argparse

parser = argparse.ArgumentParser(description="Hyper parameters")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--partitial_data", type=float, default=1.0)

args = parser.parse_args()

torch.manual_seed(42)

def main(argv):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # torch.cuda.set_device()
    device_no = 2
    device = f"cuda:{device_no}"
    train_data = SST_reader("/data/jliu/data/glue_data/SST-2/train.tsv",65)
    from torch.utils.data import Subset
    partitial_data = 0.1
    print(f"partitial_data --> {partitial_data}")
    train_data = Subset(train_data, range(int(partitial_data * len(train_data))))
    test_data=SST_reader("/data/jliu/data/glue_data/SST-2/dev.tsv",65)
    bert_name="/data/jliu/models/bert-base-uncased"
    learning_epoch=30
    
    model=SST(bert_name)
    ft_type = "LoRA"
    
    if ft_type == "FT":
        pass
    elif ft_type == "LoRA":
        flex_lora(model, 4, 8)
    elif ft_type == "adapter":
        add_adapter(model)
    elif ft_type == "our":
        customized_lora(model, 192)
        # if 
    # for layer, para in model.named_parameters():
    #     if "10" in layer or "11" in layer:
    #         if "lora" in layer:
    #             para.requires_grad = True
    #         else:
    #             para.requires_grad = False
    #     else:
    #         para.requires_grad = False
    # test same lora config diff layer
    # for layer, para in model.named_parameters():
    #     if para.requires_grad:
    #         if "out_layer" not in layer:
    #             layer_idx = int(layer.split(".")[3])
    #             # print(f"layer_idx = {layer_idx}")
    #             if layer_idx < 8:
    #                 print(f"layer -> {layer}")
    #                 para.requires_grad = False

    print(f"The model architecture --> ")
    trainable_paras = 0
    all_paras = 0

    for layer, para in model.named_parameters():
        # print(f"{layer} --> para.shape = {para.shape}")
        all_paras += para.numel()
        if para.requires_grad:
            print(f"{layer} --> para.shape = {para.shape}; para.requires_grad -> {para.requires_grad}")
            trainable_paras += para.numel()
        
    # import pdb
    # pdb.set_trace()

    print(f"Trainable paras: {trainable_paras}, all paras: {all_paras} ---> {trainable_paras / all_paras}")
    model=model.to(device)
    model.train()
    optimizer=torch.optim.SGD(model.parameters(),2e-2) # 2e-5 92.5；3e-5 91.5; 4e-5 91.1; 5e-5 90.5
    loader_train=torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    loader_test=torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=False)
    
    print("############### Before training eval ####################")
    with torch.no_grad():
        model.eval()
        loss_sum=[]
        correct=0
        total=0
        for num, (label, mask, token) in enumerate(loader_test):
            label = label.to(device)
            mask = mask.to(device)
            token = token.to(device)
            pre, loss = model(label, mask, token)
            loss_sum.append(loss.item())
            pre = torch.argmax(pre, -1)
            correct += (pre == label).sum().cpu().item()
            total += label.shape[0]
        # print("test loss:", mean(loss_sum))
        print("test accuracy: %s" % str(correct / total))

    max_acc = 0.0
    
    
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(learning_epoch):
        print(f"################# training epoch {i} ###################")
        model.train()
        loss_sum=[]
        # correct=0
        # total=0
        for num,(label,mask,token) in enumerate(loader_train):
            label=label.to(device)
            mask=mask.to(device)
            token=token.to(device)
            pre,loss=model(label,mask,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            # correct+=(pre==label).sum().cpu().item()
            # total+=label.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if num % 40 == 0:
                with torch.no_grad():
                    correct = 0
                    total = 0
                    # print('eval epoch %s\n'%str(i))
                    for num, (label, mask, token) in enumerate(loader_test):
                        label = label.to(device)
                        mask = mask.to(device)
                        token = token.to(device)
                        pre, loss = model(label, mask, token)
                        loss_sum.append(loss.item())
                        pre = torch.argmax(pre, -1)
                        correct += (pre == label).sum().cpu().item()
                        total += label.shape[0]
                    # print("test loss:", mean(loss_sum))
                    # print("test accuracy: %s" % str(correct / total))
                    max_acc = max(correct / total, max_acc)

        # print("train loss:",mean(loss_sum))
        # print("train accuracy: %s"%str(correct/total))
        model.eval()
        loss_sum=[]
        correct=0
        total=0
        with torch.no_grad():
            # print("final test")
            for num, (label, mask, token) in enumerate(loader_test):
                label = label.to(device)
                mask = mask.to(device)
                token = token.to(device)
                pre, loss = model(label, mask, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            # print("test loss:", mean(loss_sum))
            # print("test accuracy: %s" % str(correct / total))
            max_acc = max(correct / total, max_acc)
        print(f"partitial data --> {partitial_data}; max acc: {max_acc}")
    # print(f"partitial data --> {partitial_data}; max acc: {max_acc}")






def add_adapter(model, width = 32, depth = 12):
    def make_only_adapter_trainable(model):
        for layer, para in model.named_parameters():
            if "adapter" in layer:
                para.requires_grad = True
            else:
                para.requires_grad = False

    from torch import nn
    class Adapter(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super().__init__()
            self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=False) 
            self.activation = nn.ReLU()  
            self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=False)
            
        def forward(self, x):
            x = self.down_project(x) 
            x = self.activation(x)
            x = self.up_project(x)
            return x
        
    layers = [str(l) for l in range(11, 11 - depth, -1)]
    for layer in layers:
        origin_layer = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"]
        from torch.nn import Sequential
        import copy
        new_layer = Sequential()
        new_layer.add_module(layer, copy.deepcopy(origin_layer))
        adapter = Adapter(input_dim=768, bottleneck_dim=width)
        new_layer.add_module('adapter', adapter)

        model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"] = new_layer

    make_only_adapter_trainable(model)
    # 设置head可训练
    model._modules["out_layer"].weight.requires_grad = True
    model._modules["out_layer"].bias.requires_grad = True


def customized_lora(model, all_rank):
    def findMaxLowerPowerOf2(n):
        power = math.floor(math.log2(n))
        return 1 << (power - 1)

    def alg(all_rank, max_len):
        ans = list()
        while all_rank > 2:
            ans.append(findMaxLowerPowerOf2(all_rank))
            all_rank -= ans[-1]
            if len(ans) == max_len:
                return ans
        if all_rank == 2:
            ans.append(all_rank) 
        return ans
    ranks = alg(all_rank, 6)
    print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
    lora.mark_only_lora_as_trainable(model)
    
    # 设置head可训练
    model._modules["out_layer"].weight.requires_grad = True
    model._modules["out_layer"].bias.requires_grad = True
    
    return model


def focused_lora(model, all_rank, depth,target_matrix):
    def my_alg(num, parts):
        res = []
        while num > 0:
            if len(res) < parts - 1:
                cur = int(num / 2)
                res.append(cur)
                num -= cur
            else:
                res.append(num)
                break
            # print(res)
        return res

    ranks = my_alg(all_rank, depth)
    print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
    lora.mark_only_lora_as_trainable(model)
    
    # 设置head可训练
    model._modules["out_layer"].weight.requires_grad = True
    model._modules["out_layer"].bias.requires_grad = True
    
    return model



def make_focused_layer(layer_rank, target_matrix):
    rank_distri = {
        "query": layer_rank, 
        "key": layer_rank, 
        "value": layer_rank, 
        "output": layer_rank,
        "intermediate": layer_rank, 
        "output": layer_rank
    }

    max_rank = layer_rank * 6
    for matrix in rank_distri.keys():
        if matrix == target_matrix:
            rank_distri[matrix] = max_rank
        else:
            rank_distri[matrix] = 1
        

        



def flex_lora(model, rank = 5, alpha = 10):
    ranks = [16 for _ in range(12)]
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
    lora.mark_only_lora_as_trainable(model)
    
    # 设置head可训练
    model._modules["out_layer"].weight.requires_grad = True
    model._modules["out_layer"].bias.requires_grad = True
    
    return model


if __name__ == '__main__':
    app.run(main)
