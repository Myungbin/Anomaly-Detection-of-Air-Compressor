def huber_loss(y_pred, y_true, delta=1.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(loss)