"""
desafio2_gustavo_pablo.py — tanh com inicializacao gaussiana calibrada 
(s^2 = 1/(fan_in*E[tanh(z)^2]), Monte Carlo, vies zero, metade na camada de logits), 
e o desvio de TODAS as camadas multiplicado por 0.01 ** (1/(2*n_camadas)). 
Isso distribui um "orcamento" fixo de amortecimento total 
(fator 0.01 acumulado da entrada ate a saida) igualmente
em log-espaco entre as n_camadas camadas, seja qual for a profundidade.
"""
import math
import torch


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        desvio *= 0.5
    desvio *= 0.01 ** (1.0 / (2 * n_camadas))
    W.normal_(0.0, desvio)
    b.zero_()
