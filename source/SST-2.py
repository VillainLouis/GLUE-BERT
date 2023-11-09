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



def main(argv):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # torch.cuda.set_device()
    device_no = 2
    device = f"cuda:{device_no}"
    train_data = SST_reader("/data/jliu/data/glue_data/SST-2/train.tsv",65)
    test_data=SST_reader("/data/jliu/data/glue_data/SST-2/dev.tsv",65)
    bert_name="/data/jliu/models/bert-base-uncased"
    learning_epoch=3
    
    model=SST(bert_name)
    
    ft_type = "LoRA"
    
    if ft_type == "FT":
        pass
    elif ft_type == "LoRA":
        flex_lora(model)
        
    
    print(f"The model architecture --> ")
    for layer, para in model.named_parameters():
        print(f"{layer} --> para.required_grad = {para.requires_grad}")
    model=model.to(device)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),2e-5) # 2e-5 92.5；3e-5 91.5; 4e-5 91.1; 5e-5 90.5
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
        print("loss:", mean(loss_sum))
        print("accuracy: %s" % str(correct / total))
    
    
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(learning_epoch):
        print(f"################# training epoch {i} ###################")
        model.train()
        loss_sum=[]
        correct=0
        total=0
        for num,(label,mask,token) in tqdm(enumerate(loader_train)):
            label=label.to(device)
            mask=mask.to(device)
            token=token.to(device)
            pre,loss=model(label,mask,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            correct+=(pre==label).sum().cpu().item()
            total+=label.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("loss:",mean(loss_sum))
        print("accuracy: %s"%str(correct/total))
        model.eval()
        loss_sum=[]
        correct=0
        total=0
        with torch.no_grad():
            print('eval epoch %s\n'%str(i))
            for num, (label, mask, token) in enumerate(loader_test):
                label = label.to(device)
                mask = mask.to(device)
                token = token.to(device)
                pre, loss = model(label, mask, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            print("loss:", mean(loss_sum))
            print("accuracy: %s" % str(correct / total))


def flex_lora(model, rank = 8, alpha = 32):
    target_attn_matrix = { # attn
            # "0": ["query", "key", "value"],
            # "1": ["query", "key", "value"],
            # "2": ["query", "key", "value"],
            # "3": ["query", "key", "value"],
            # "4": ["query", "key", "value"],
            "5": ["query", "key", "value"],
            "6": ["query", "key", "value"],
            "7": ["query", "key", "value"],
            "8": ["query", "key", "value"],
            "9": ["query", "key", "value"],
            "10": ["query", "key", "value"],
            "11": ["query", "key", "value"]
        }
    target_ffn_matrix = { # ffn
        # "0": ["intermediate", "output"],
        # "1": ["intermediate", "output"],
        # "2": ["intermediate", "output"],
        # "3": ["intermediate", "output"],
        # "4": ["intermediate", "output"],
        "5": ["intermediate", "output"],
        "6": ["intermediate", "output"],
        "7": ["intermediate", "output"],
        "8": ["intermediate", "output"],
        "9": ["intermediate", "output"],
        "10": ["intermediate", "output"],
        "11": ["intermediate", "output"]
    }
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            module = model._modules["Bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
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
