
import torch
import my_transformer

device='cuda:0' if torch.cuda.is_available() else 'cpu'
if __name__=='__main__':
    model=my_transformer.Transformer(my_transformer.n_embed,my_transformer.n_head,my_transformer.num_layer)
    tokenizer=my_transformer.tokenizer
    #直接加载模型框架也需要有模型架构的原定义代码，transformer库用的是json配置文件保存了模型架构然后在load的时候读取配置文件重新新建一个模型架构
    model.load_state_dict(torch.load('model_state_dict.bin'))
    num_of_para=sum(p.numel() for p in model.parameters())
    print(f'Total parameters:{num_of_para}')
    model.to(device)
    model.eval()

    # for param_tensor in model.state_dict():
    #     print(param_tensor,'\t',model.state_dict()[param_tensor].device)
    #decoder初始输入是<SOW>
    with torch.device(device):
        trg_id=torch.ones((1,1),dtype=torch.long,device=device)
        # src_id=torch.cat([torch.tensor([[1]]),torch.tensor([tokenizer.encode('我的英雄学院')])],dim=-1)
        src_id=torch.tensor([tokenizer.encode('我的英雄学院')])
        # trg_id=src_id[:,0].view(-1,1)
        print(src_id)
        src_id.to(device)
        print(tokenizer.decode(model.generate(src_id,trg_id,1000)[0].tolist()))


    
