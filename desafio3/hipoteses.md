# Desafio 3 — hipóteses registradas antes das varreduras

Escritas depois da baseline de rascunho (Adam, η = 1e-4, constante, 20.000 passos,
semente 0) e **antes** de qualquer varredura de η ou de agenda. O commit deste
arquivo marca o momento. Cada hipótese diz o que a confirmaria e o que a refutaria.
Elas não serão editadas depois dos resultados; o desfecho vai numa seção separada,
no fim.

## O que a baseline de rascunho mostrou (base das previsões)

- Validação: 0,497 → mínimo 0,164 (passo 14.250) → 0,167 no fim. Chega a 1% do
  mínimo no passo ~10.500; depois só cresce a distância treino–validação
  (Huber no lote fixo do treino: 0,150 → 0,130).
- ‖g‖ entre 0,3 e 5,5 e nunca cortado (clip 10), sem picos no início.
- ‖θ‖ cresce sem parar, de 8 para 15 (wd = 0).
- ρ mediano (passos ≥ 2000): conv1 2,9e-4, conv2 4,5e-4, conv3 3,7e-4,
  cabeca_q 2,1e-4, cabeca_esperar 9,6e-5. Tudo abaixo da referência de 1e-3 do enunciado.
- 8–11% dos filtros da conv3 nunca ativam no lote fixo; nenhum na conv1 nem na conv2.

## Hipóteses

### H1. O melhor η do Adam fica acima de 1e-4
Os ρ da baseline (≈ 3–5e-4 nas convs) estão abaixo da referência de ~1e-3: o passo
é pequeno. **Confirma:** na varredura com warmup + cosseno, o menor val_final do Adam
fica em η > 1e-4. **Refuta:** o melhor η é ≤ 1e-4.

### H2. AdamW (weight decay) estabiliza ‖θ‖ e reduz a distância treino–validação
Na baseline, ‖θ‖ cresce linearmente e a distância treino–validação abre depois do
passo ~9000. **Confirma:** no AdamW, com o melhor η dele, ‖θ‖ se estabiliza (ou cresce
bem menos) e a diferença huber_val − huber_treino no fim é menor que a do Adam no
mesmo η. **Refuta:** ‖θ‖ continua crescendo no mesmo ritmo, ou a distância não diminui.

### H3. O warmup ajuda pouco no Adam
Não há picos iniciais em ‖g‖ na baseline, que é o sintoma que o warmup corrige.
**Confirma:** no Adam, com o mesmo η, warmup_cosseno e cosseno diferem menos que a
variação entre sementes. **Refuta:** o warmup melhora o val_final além dessa variação,
ou evita uma divergência que acontece sem ele.

### H4. Boa parte da perda de validação (~0,16) é ruído irredutível; as diferenças entre otimizadores vão ser pequenas
O alvo é o retorno de Monte Carlo de uma única partida. Estados iguais (ou quase) com
desfechos diferentes recebem G diferente, que pode diferir em ~2 entre vitória e
derrota. No subconjunto de 1000 amostras da verificação de sobreajuste, só os estados
idênticos já formam um piso de 1,6% da perda inicial. No dataset inteiro, estados
parecidos com G diferente devem pesar muito mais. **Confirma:** os val_final dos
melhores (otimizador, η) de cada família ficam a poucos % uns dos outros, da mesma
ordem da variação entre sementes. **Refuta:** algum otimizador fica claramente abaixo
dos outros, por bem mais que a variação entre sementes.

### H5. O ρ menor da cabeca_esperar não vem de "esperar" ser rara
**Fato medido:** "esperar" é **81,8%** das ações do treino (79,6% nas partidas
vencidas, 84,8% nas perdidas) e 81,7% na validação. A cabeca_esperar recebe gradiente
em ~4 de cada 5 amostras. Mesmo assim, o ρ dela é 2–5× menor que o das outras camadas
(9,6e-5 contra 2,1e-4 a 4,5e-4).
**Hipótese:** a causa é o tamanho e o sinal, não a frequência. São 65 parâmetros que
recebem a média global de 64 canais, e o alvo de esperar é ruidoso: o mesmo estado
pode virar vitória ou derrota. No Adam, gradientes de sinal instável dão m/√v pequeno,
então o passo efetivo fica bem abaixo de η.
**Confirma:** a razão ρ_cabeca_esperar / ρ_conv se mantém parecida entre Adam, AdamW e
RMSprop, e muda nos SGDs (que não normalizam por √v).
**Refuta:** a razão é a mesma em todos os otimizadores.

## Acréscimos do passo 0 (antes das varreduras)

### Critério de vitória
A vence B se a diferença das médias de val_final (5 sementes cada) passar de 2 desvios-padrão
combinados: |média_A − média_B| > 2·s_c, com s_c = √((s_A² + s_B²)/2) (desvio-padrão
combinado de amostras do mesmo tamanho; s_A e s_B são os desvios amostrais, n − 1).
O lado vencedor é o de menor val_final.

### Validação dividida
A partir daqui também se registram val_esperar (Huber só nas amostras de validação cuja
ação foi ESPERAR, ~82%) e val_plantio (só nas demais, ~18%). A métrica de seleção
continua sendo val_final.

### H6. Se o Muon ganhar, o ganho aparece mais em val_plantio do que em val_esperar
O sinal de plantio é raro: 18% das amostras, espalhadas em 225 células do mapa Q. O
enunciado afirma que o Muon dá o mesmo tamanho de passo a todas as direções singulares,
inclusive às raras, que o SGD e o Adam quase ignoram. **Confirma:** o Muon vence pelo
critério acima, e a redução relativa de val_plantio (contra a baseline e contra o
melhor não-Muon) é maior que a de val_esperar. **Refuta:** o Muon vence, mas o ganho
relativo é igual ou maior em val_esperar. (Se o Muon não vencer, H6 não se aplica.)

### Confirmação: 4ª combinação extra
Além das 3 exigidas pelo protocolo (as 3 melhores do passo 3: sgd 3e-2 warmup_cosseno,
sgd 3e-2 cosseno e rmsprop 3e-4 warmup_cosseno), a confirmação inclui, **além das 3
exigidas pelo protocolo**, o melhor Muon do passo 3 (η das matrizes 1e-3, AdamW 3e-4,
warmup_cosseno) com as sementes 0–4. Motivo: o Muon é obrigatório, o painel pede "a
melhor com Muon" e a H6 precisa de barra de erro. Decidido depois do passo 3 e antes de
rodar a confirmação.

## Extras (além do protocolo), registrados antes de rodar
O protocolo terminou no passo 4. Os extras abaixo **não mudam a vencedora**: se algum
tiver média melhor, isso só é relatado.

### E1. Adam 3e-4 e AdamW 3e-4 (warmup_cosseno, 5 sementes) empatam com as finalistas
Na varredura (1 semente) eles ficaram a 0,0005–0,0013 do RMSprop, dentro da variação
entre sementes vista no passo 4 (s ≈ 0,0014–0,0022). **Previsão:** pelo critério de
vitória, nenhum dos dois vence nem perde para qualquer das 3 finalistas.

### E2. A rede jogando (RedeGulosa) perde para o Econ5
A RedeGulosa escolhe argmax_a Q(s, a) entre as ações válidas, usando só observar() e
mascara(). **Previsão:** ela vence menos partidas que o Econ5 (sem aleatoriedade) nas
mesmas fases e sementes. A rede aprendeu o valor das jogadas *do Econ5* (com 15% de
ruído), não o das jogadas ótimas. Como 82% dos dados são "esperar", as jogadas raras
estão mal estimadas, e o argmax vai atrás de erros otimistas ou espera quase sempre.

## Vencedora declarada antes do teste
Vencedora declarada antes do teste: SGD 3e-2 cosseno (melhor média de val_final no passo 4;
desempate pela receita mais simples, sem warmup). Avaliadas no teste: vencedora, baseline e
Muon 1e-3 warmup_cosseno.

## Desfecho

Escrito depois das etapas 2–4, do teste e dos extras. As hipóteses acima não foram editadas.
Números: `desafio3/tabelas.md`, `desafio3/teste.csv` e os logs em `desafio3/logs/`.
Médias ± desvio amostral nas 5 sementes da confirmação, salvo indicação.

### H1. O melhor η do Adam fica acima de 1e-4: **confirmada**
Na varredura com warmup + cosseno, o menor val_final do Adam foi com η = 3e-4, acima de 1e-4.
O mesmo vale para o AdamW e o RMSprop (3e-4). Com 3e-4, o ρ das convoluções sobe para perto da
referência de 1e-3 no começo do treino.

### H2. O AdamW estabiliza ‖θ‖ e reduz a distância treino–validação: **refutada**
No mesmo η (3e-4, warmup + cosseno), AdamW e Adam terminam praticamente iguais: ‖θ‖ = 14,61
contra 14,59, e huber_val − huber_treino no fim = 0,034 contra 0,035. O motivo é a escala do
decay desacoplado: a cada passo os pesos encolhem η·λ = 3e-4 × 0,01 = 3·10⁻⁶; em 10.000 passos
isso dá ~3%, pouco diante do crescimento de 8 para 14,6. O ‖θ‖ para de crescer no fim nos dois,
e quem faz isso é o cosseno levando η a zero, não o weight decay.

### H3. O warmup ajuda pouco: **confirmada no SGD** (não testada diretamente no Adam)
A etapa de agendas rodou com os 2 melhores otimizadores (SGD e RMSprop) e o Muon; o Adam não
entrou. No SGD 3e-2, com e sem warmup: 0,16227 ± 0,00189 contra 0,16225 ± 0,00219, diferença
muito menor que a variação entre sementes. Coerente com o sintoma ausente: ‖g‖ nunca teve
picos no começo e nunca chegou ao corte (10).

### H4. As diferenças entre otimizadores vão ser pequenas: **confirmada em parte**
Em termos relativos, pequenas: o melhor val_final de cada família fica a no máximo 2,4% do melhor
(SGD 0,1623; RMSprop 0,1623; AdamW 0,1656; Adam 0,1661), e as 3 finalistas diferem em 0,0001.
Mas algumas diferenças passam do critério: o Adam 3e-4 perde para as 3 finalistas (2,0–2,5 s_c),
e as finalistas vencem a baseline (3,2 s_c). O Muon fica fora do padrão, 7,8% pior.

### H5. O ρ menor da cabeça de "esperar" vem do tamanho e do sinal, não da frequência: **confirmada**
Razão ρ_cabeca_esperar / ρ_conv (medianas a partir do passo 2.000, média das execuções):
Adam 1e-4 0,29; Adam 3e-4 0,27; AdamW 3e-4 0,26; RMSprop 3e-4 0,30 nos adaptativos, contra
SGD 3e-2 0,51–0,52 (Nesterov, 1 semente: 0,38). A razão é parecida entre os que normalizam por
√v e muda nos SGDs, como previsto. Ela continua abaixo de 1 no SGD, então o tamanho da cabeça
também pesa, não só o sinal instável.

### H6. Se o Muon vencer, o ganho aparece mais em val_plantio: **não se aplica**
O Muon não venceu: 0,1750 ± 0,0003, pior que a baseline (0,1676) e que todas as finalistas.
Ele teve a menor perda de treino (0,111) e a pior validação, com o mínimo da validação por
volta do passo 3.000–4.000: decorou o ruído do alvo.

### E1. Adam 3e-4 e AdamW 3e-4 empatam com as finalistas: **refutada em parte**
O Adam 3e-4 perde para as 3 finalistas (+2,0, +2,2 e +2,5 s_c). O AdamW 3e-4 empata com as duas
do SGD (+1,8 e +1,9 s_c) e perde para o RMSprop (+2,3 s_c).

### E2. A rede jogando (RedeGulosa) perde para o Econ5: **confirmada**
Em 420 partidas (21 fases de treino × sementes 200–219), a RedeGulosa venceu 0,5% (vencedora)
e 0,0% (baseline), contra 64,5% do Econ5. A rede joga ~85 decisões por partida e planta
girassóis e minas, quase nenhum atirador: o argmax vai atrás de erros nas jogadas raras.

### Teste (usado uma vez)
Vencedora (SGD 3e-2 cosseno) 0,1344 ± 0,0034; baseline 0,1394 ± 0,0013; Muon 0,1438 ± 0,0017.
A ordem da validação se mantém. Vencedora × baseline: Δ = −0,0050 com 2·s_c = 0,0052 (1,9 s_c),
empate formal no teste, no mesmo sentido da validação. Muon × baseline: Δ = +0,0044 com
2·s_c = 0,0031: o Muon perde também no teste.
