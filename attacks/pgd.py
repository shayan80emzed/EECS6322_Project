import torch

def pgd(
    forward,
    loss_fn,
    data_clean,
    targets,
    norm='linf',
    eps=4/255,
    iterations=10,
    stepsize=1/255,
    output_normalize=False,
    momentum=0.9
):
    if norm.lower() not in ['linf', 'inf']:
        raise ValueError(f"Only 'linf' norm is supported in this PGD implementation. Got '{norm}'.")

    perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)

    for _ in range(iterations):
        adv_inputs = (data_clean + perturbation).clamp(0, 1).detach()
        adv_inputs.requires_grad = True

        with torch.enable_grad():
            outputs = forward(adv_inputs, output_normalize=output_normalize)
            loss = loss_fn(outputs, targets)

        grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
        grad = grad.sign() # linf norm

        velocity = momentum * velocity + grad
        velocity = velocity.sign()

        perturbation = perturbation + stepsize * velocity
        perturbation = torch.clamp(perturbation, -eps, eps) # linf norm
        perturbation = (data_clean + perturbation).clamp(0, 1) - data_clean

    return (data_clean + perturbation).detach().clamp(0, 1)
