# Desafio 3 — O melhor treino para o nosso problema

GBC073 Inteligência Computacional (FACOM/UFU) · Gustavo Luis de Siqueira Nascimento e
Pablo Rodrigues Cardoso Araújo.

Cópia da entrega do Desafio 3, tirada do repositório do projeto (simulador de Plants vs.
Zombies próprio + agente de RL), branch `main`, commit `7e17406`.

## O que está aqui

| caminho | o que é |
|---|---|
| `desafio3/treino.py` | treino configurável (otimizador, η, agenda, semente) com o registro dos sinais |
| `desafio3/experimentos.csv` | uma linha por execução: configuração, val_final, divergiu?, tempo (79 execuções) |
| `desafio3/logs/` | logs de todas as execuções (config.json e registro.csv) |
| `desafio3/painel.pdf` | os quatro sinais alinhados (melhor, baseline, melhor Muon, instável) e a métrica × η |
| `desafio3/slide.pdf` | a apresentação |
| `desafio3/hipoteses.md` | hipóteses registradas antes das varreduras, e o desfecho |
| `desafio3/tabelas.md` | tabelas finais; `desafio3/teste.csv`: o teste, avaliado uma vez |
| `desafio3/muon.py` | Muon próprio (Newton–Schulz, 5 iterações; kernels achatados) |
| `desafio3/dados.py` | gera o dataset do problema-proxy (os arrays não são versionados; `dados/meta.json` tem os hashes) |
| `pvz/` | o simulador do jogo, necessário para gerar o dataset |
| `tests/test_desafio3.py` | testes da CNN, dos registries, das agendas e do Muon |

## Reproduzir uma linha do experimentos.csv

```
pip install -r requirements.txt
python -m desafio3.dados                     # gera o dataset (confira os hashes no meta.json)
python -m desafio3.treino --otimizador sgd --lr 3e-2 --agenda cosseno --semente 0 --passos 10000 --etapa confirmacao --rascunho
python -m pytest tests/test_desafio3.py
```

O treino recusa rodar com alterações não commitadas sem `--rascunho`; rascunhos vão para
`desafio3/logs/rascunho/` e não entram no `experimentos.csv`. Fora do repositório original, o
hash de commit gravado nos logs será outro.
