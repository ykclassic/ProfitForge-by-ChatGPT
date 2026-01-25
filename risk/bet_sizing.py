def fractional_kelly(p, avg_win, avg_loss, fraction=0.25):
    b = avg_win / avg_loss
    kelly = (p * b - (1 - p)) / b
    return max(0.0, kelly * fraction)
