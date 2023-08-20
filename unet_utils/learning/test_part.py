import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from unet_utils.data.load_data import create_data_loaders
from unet_utils.model.ResUnet import ResUnet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    with torch.no_grad():
        for i, (input_, _,maximum,fnames, slices) in enumerate(data_loader):
#             if i>5:
#                 break
            if i % 100 == 0:
                print(f'Saved {i} images')
            
            input_, target, maximum, fnames, slices = data
            if args.is_grappa != 'y' and args.given_grappa != 'y':
                input_ = input_.unsqueeze(0)
            input_ = input_.cuda(non_blocking=True)
            
            
            output = model(input_)

            for i in range(1):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = ResUnet(args.chanels)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    import pdb; pdb.set_trace()
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, mode=None, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)