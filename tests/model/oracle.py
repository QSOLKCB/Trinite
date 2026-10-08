"""Small scalar-math oracle: no torch imports or implementation calls."""
import math
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def effective(matrix, lane):
    if lane == "dense":
        return matrix
    scale = f32(max(math.fsum(abs(v) for row in matrix for v in row)
                    / sum(len(row) for row in matrix), 1e-8))
    return [[scale * max(-1, min(1, round(f32(v/scale)))) for v in row] for row in matrix]


def mat(value, weight):
    return [math.fsum(a*b for a, b in zip(value, row)) for row in weight]


def norm(value, gain):
    denominator = math.sqrt(math.fsum(x*x for x in value)/len(value) + 1e-5)
    return [x/denominator*g for x, g in zip(value, gain)]


def forward(parameters, ids, mask, lane):
    """One block, width four, two heads; operation order from the contract."""
    token, position = parameters["token_embedding"], parameters["position_embedding"]
    value = [[a+b if live else 0.0 for a, b in zip(token[t], position[i])]
             for i, (t, live) in enumerate(zip(ids, mask))]
    captures = {"embedding": value}
    p = "blocks.0."
    normalized = [norm(v, parameters[p+"attention_norm.weight"]) for v in value]
    q, k, v = [[mat(x, effective(parameters[p+n+".weight"], lane)) for x in normalized]
               for n in ("q", "k", "v")]
    attended = []
    for i in range(len(ids)):
        row = []
        for h in range(2):
            scores = [math.fsum(q[i][h*2+d] * k[j][h*2+d] for d in range(2))/math.sqrt(2)
                      for j in range(i+1) if mask[j]]
            highest = max(scores)
            numerators = [math.exp(s-highest) for s in scores]
            denominator = math.fsum(numerators)
            live_positions = [j for j in range(i+1) if mask[j]]
            row.extend(math.fsum(n/denominator * v[j][h*2+d]
                                 for n, j in zip(numerators, live_positions)) for d in range(2))
        attended.append(row)
    residual = [[a+b if live else 0.0 for a, b in zip(old, mat(attn, effective(parameters[p+"o.weight"], lane)))]
                for old, attn, live in zip(value, attended, mask)]
    value = []
    for old, live in zip(residual, mask):
        up = mat(norm(old, parameters[p+"feed_forward_norm.weight"]),
                 effective(parameters[p+"up.weight"], lane))
        gelu = [0.5*x*(1+math.erf(x/math.sqrt(2))) for x in up]
        down = mat(gelu, effective(parameters[p+"down.weight"], lane))
        value.append([a+b if live else 0.0 for a, b in zip(old, down)])
    captures["block.0"] = value
    value = [norm(x, parameters["final_norm.weight"]) for x in value]
    captures["final"] = value
    return [mat(x, token) for x in value], captures


def hand_parameters():
    shapes = [("token_embedding", (259, 4)), ("position_embedding", (8, 4)),
              ("blocks.0.attention_norm.weight", (4,)),
              *[("blocks.0."+n+".weight", (4, 4)) for n in ("q", "k", "v", "o")],
              ("blocks.0.feed_forward_norm.weight", (4,)),
              ("blocks.0.up.weight", (8, 4)), ("blocks.0.down.weight", (4, 8)),
              ("final_norm.weight", (4,))]
    result = {}
    for offset, (name, shape) in enumerate(shapes):
        values = [((i+offset) % 17 - 8)/32 for i in range(math.prod(shape))]
        if len(shape) == 1:
            result[name] = [1+x/8 for x in values]
        else:
            result[name] = [values[i:i+shape[1]] for i in range(0, len(values), shape[1])]
    return result
