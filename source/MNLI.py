from MNLI.MNLI_reader import MNLI_reader
from MNLI.MNLI_model import MNLI
import torch
import transformers
from absl import app, flags
import os
from tqdm import tqdm
import torch
from numpy import mean


def main(argv):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # torch.cuda.set_device()
    device_no = 2
    device = f"cuda:{device_no}"
    train_data = MNLI_reader("/data/jliu/data/glue_data/MNLI/train.tsv",200)
    test_data_mis=MNLI_reader("/data/jliu/data/glue_data/MNLI/dev_mismatched.tsv",200)
    test_data_ma=MNLI_reader("/data/jliu/data/glue_data/MNLI/dev_matched.tsv",200)
    bert_name="/data/jliu/models/bert-base-uncased"
    learning_epoch=10
    model=MNLI(bert_name,200,768)
    model=model.to(device)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),5e-5)
    loader_train=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    model.zero_grad()
    optimizer.zero_grad()
    for i in range(learning_epoch):
        print("training epoch ",i)
        loss_sum=[]
        correct=0
        total=0
        for num,(label,mask,segment,token) in tqdm(enumerate(loader_train)):
            label=label.to(device)
            mask=mask.to(device)
            token=token.to(device)
            segment=segment.to(device)
            pre,loss=model(label,mask,segment,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            correct+=(pre==label).sum().cpu().item()
            total+=label.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("loss:",mean(loss_sum))
        print("accuracy:%s",str(correct/total))
        loader_test_misma=torch.utils.data.DataLoader(test_data_mis,batch_size=32,shuffle=False)
        loader_test_match=torch.utils.data.DataLoader(test_data_ma,batch_size=32,shuffle=False)
        model.eval()
        loss_sum=[]
        with torch.no_grad():
            print('eval mismatch epoch %s\n'%str(i))
            correct = 0
            total = 0
            for num, (label,mask,segment, token) in tqdm(enumerate(loader_test_misma)):
                label = label.to(device)
                mask = mask.to(device)
                segment = segment.to(device)
                token = token.to(device)
                pre, loss = model(label, mask,segment, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            print("mismatch loss:", mean(loss_sum))
            print("mismatch accuracy:%s", str(correct / total))
            print('mismatch eval match epoch %s\n'%str(i))
            correct = 0
            total = 0
            for num, (label,mask,segment, token) in tqdm(enumerate(loader_test_match)):
                label = label.to(device)
                mask = mask.to(device)
                segment = segment.to(device)
                token = token.to(device)
                pre, loss = model(label, mask,segment, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            print("match loss:", mean(loss_sum))
            print("match accuracy:%s", str(correct / total))


if __name__ == '__main__':
    app.run(main)
