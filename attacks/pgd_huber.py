import torch

def pgd(
    forward,
    loss_fn,
    data_clean,
    targets,
    norm='huber',
    eps=4/255,
    eps2=4/255,
    iterations=10,
    stepsize=1/255,
    output_normalize=False,
    momentum=0.9,
):
    if norm.lower() not in ['linf', 'inf', 'huber']:
        raise ValueError(f"Only 'linf' and 'huber' norms are supported. Received: '{norm}'")

    perturbation = torch.zeros_like(data_clean, requires_grad=True)
    velocity = torch.zeros_like(data_clean)

    for _ in range(iterations):
        noisy_inputs = (data_clean + torch.randn_like(data_clean) / 255).clamp(0, 1)    # gaussian noise

        perturbed_inputs = (noisy_inputs + perturbation).clamp(0, 1).detach()
        perturbed_inputs.requires_grad = True

        with torch.enable_grad():
            outputs = forward(perturbed_inputs, output_normalize=output_normalize)
            loss = loss_fn(outputs, targets)

            # huber
            if norm.lower() == 'huber' and eps2 > 0:
                pert_norm = perturbation.view(perturbation.size(0), -1).norm(p=2, dim=1).mean()
                loss -= eps2 / eps * pert_norm

        grad = torch.autograd.grad(loss, perturbed_inputs, retain_graph=False, create_graph=False)[0]
        grad = grad.sign()

        velocity = momentum * velocity + grad
        velocity = velocity.sign()

        perturbation = perturbation + stepsize * velocity
        perturbation = torch.clamp(perturbation, -eps, eps)

        perturbation = (data_clean + perturbation).clamp(0, 1) - data_clean

    return (data_clean + perturbation).detach().clamp(0, 1)
