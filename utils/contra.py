import torch
import logging
        
def cast_to_range(values, scale):
    return torch.round(values * scale).to(torch.long) 

def uncast_from_range(scaled_values, scale):
    return scaled_values / scale

def cal_similarity(x, y):
    # Ensuring no division by zero
    x[torch.abs(x) < 1e-5] = 1e-5
    y[torch.abs(y) < 1e-5] = 1e-5

    # Calculating max(x_i/y_i, y_i/x_i) for each element
    max_ratios = torch.max(x / y, y / x)

    # Calculating min(max(x_i/y_i, y_i/x_i)) and max(max(x_i/y_i, y_i/x_i))
    min_of_max_ratios = torch.min(max_ratios)
    max_of_max_ratios = torch.max(max_ratios)

    # Calculating the final ratio
    final_ratio = min_of_max_ratios / max_of_max_ratios
    logging.info(final_ratio)
    return final_ratio


def modinv(a, m):
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, x, y = egcd(b % a, a)
            return (g, y - (b // a) * x, x)

    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def contra(args):
        for i in range(args.num_clients):
            # a_inv = torch.pow(args.a[i], torch.reciprocal(args.p))
            a_inv = modinv(args.a[i], args.p)
            # g_inv = torch.pow(args.g, torch.reciprocal(args.p))
            # last_layer = ((args.cos_client_ref[i] * a_inv * g_inv) % args.p) / (args.a[i] * args.g)
            # a_inv = modinv(args.a[i], args.p)
            # g_inv = modinv(args.g, args.p)
            # last_layer = ((args.cos_client_ref[i] * a_inv) % args.p * g_inv) % args.p
            last_layer = (args.cos_client_ref[i] * a_inv) % args.p
            last_layer[last_layer > args.p // 2] -= args.p      
            last_layer = uncast_from_range(last_layer, args.g)
            if i == 5:
                logging.info('after'  + '-' * 64) 
                logging.info(torch.max(last_layer))
                logging.info(torch.min(last_layer))
            # last_layer /= args.p 
            # last_layer = last_layer / args.g
            if torch.linalg.norm(last_layer) > 1:
                last_layer /= torch.linalg.norm(last_layer) 
            args.cos_client_ref0[i] += last_layer
        cos_client_ref = args.cos_client_ref0
        epsilon = 1e-9
        n = len(cos_client_ref)
        args.client_learning_rate = torch.ones(len(cos_client_ref)) 

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        tao = torch.zeros(n)
        topk = n // 5
        t = 0.5
        delta = 0.1

        cs = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(cos_client_ref[i].float(), cos_client_ref[j].float()).item()
                # cs[i][j] = cal_similarity(cos_client_ref[i], cos_client_ref[j]).item()

        maxcs = torch.max(cs, dim = 1).values + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        for i in range(n):
            tao[i] = torch.mean(torch.topk(cs[i], topk).values)
        #     if tao[i] > t:
        #         client_reputation[i] -= delta 
        #     else:
        #         client_reputation[i] += delta 

        #  Pardoning: reweight by the max value seen
        args.client_learning_rate = torch.ones((n)) - tao
        args.client_learning_rate /= torch.max(args.client_learning_rate)
        args.client_learning_rate[args.client_learning_rate==1] = 0.99
        args.client_learning_rate = (torch.log((args.client_learning_rate / (1 - args.client_learning_rate)) + epsilon) + 0.5)
        args.client_learning_rate[(torch.isinf(args.client_learning_rate) + args.client_learning_rate > 1)] = 1
        args.client_learning_rate[(args.client_learning_rate < 0)] = 0
        args.client_learning_rate /= torch.sum(args.client_learning_rate)
        args.client_learning_rate = cast_to_range(args.client_learning_rate, args.w)