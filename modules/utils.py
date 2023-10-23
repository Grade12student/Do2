import torch


# Utility Functions

def generate_noise(size, device):
    noise = torch.randn(size).to(device)
    return noise


def get_scales_by_index(index, scale_factor, stop_scale, img_size):
    size = img_size
    for i in range(index):
        size = int(size * scale_factor)
        if size == stop_scale:
            break
    return size


def adjust_scales2image(img_size, opt):
    opt.stop_scale = opt.min_size
    opt.scale_weights = []
    opt.scale_sizes = []
    size = opt.img_size
    opt.stop_scale = min(opt.max_size, img_size)
    opt.scale_sizes.append((size, size))
    opt.scale_weights.append(1)
    while size > opt.stop_scale:
        size = int(size * opt.scale_factor)
        opt.scale_sizes.append((size, size))
        opt.scale_weights.append(opt.scale_weights[-1] * 2)

    opt.scale_sizes = list(reversed(opt.scale_sizes))
    opt.scale_weights = list(reversed(opt.scale_weights))


# Main Function

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# Additional Utility Functions
# Add any additional utility functions or custom transformations here

# ...

# Add any custom preprocessing or evaluation functions here

# ...

# Add any custom model weight initialization or optimizer functions here

# ...

