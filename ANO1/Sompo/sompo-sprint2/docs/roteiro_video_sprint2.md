# 🎬 Roteiro do Vídeo — Sprint 2

> Duração total: **5 minutos** (limite do briefing)
> Apresentadores: **Renata** (PO/UX) e **João** (Tech Lead)
> Formato: Loom / OBS — gravação de tela + voz
> Configuração de privacidade: **não-listado** no YouTube

---

## Estrutura — 6 slides em 5 minutos

| Slide | Tempo | Acumulado | Quem fala |
|---|---|---|---|
| 1. Recap da Sprint 1 + o que mudou | 0:30 | 0:30 | Renata |
| 2. User Stories formalizadas (resposta ao feedback) | 0:45 | 1:15 | Renata |
| 3. Dataset v2 ampliado | 0:40 | 1:55 | João |
| 4. Modelo treinado + métricas | 1:00 | 2:55 | João |
| 5. Demo do dashboard | 1:15 | 4:10 | Renata |
| 6. Arquitetura consolidada + roadmap | 0:40 | 4:50 | João |
| (Fechamento) | 0:10 | 5:00 | Os dois |

---

## SLIDE 1 — Recap da Sprint 1 + o que mudou (30s)

**No slide:**
- Título: **Sompo AgroPredict — Sprint 2**
- Subtítulo: *Do reativo ao preditivo*
- Linha do tempo: Sprint 1 ✅ → **Sprint 2 ✅** → Sprint 3 → Sprint Final
- Lista compacta do que a Sprint 2 entrega: Modelo treinado · Camada SQL · Dashboard funcional · User Stories

**Fala (Renata):**
> "Oi, gente! Sou a Renata e esse é o João, do Grupo 30. Estamos voltando pra apresentar a Sprint 2 do Sompo AgroPredict.
>
> Na Sprint 1 a gente entregou a proposta documentada — problema, solução, personas e arquitetura. Agora na Sprint 2 a gente transformou tudo isso em **código funcionando**: modelo treinado, banco de dados no Oracle, dashboard rodando, e formalizamos as User Stories que a tutora pediu no feedback."

---

## SLIDE 2 — User Stories (resposta direta ao feedback) (45s)

**No slide:**
- Título: **User Stories — Resposta ao feedback**
- 3 cards lado a lado, um por persona:

```
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ US-01          │ │ US-02          │ │ US-03          │
│ Operador       │ │ Gestor         │ │ Analista Sompo │
│                │ │                │ │                │
│ Alerta em      │ │ Dashboard      │ │ Relatórios de  │
│ tempo real     │ │ tático diário  │ │ comportamento  │
└────────────────┘ └────────────────┘ └────────────────┘
```

- Embaixo: exemplo de critério de aceite formatado

```
DADO solo saturado + chuva recente
QUANDO o modelo avalia a operação
ENTÃO exibe ALTO em tela cheia + alerta sonoro
  E registra em predicoes E cria entrada em alertas
```

**Fala (Renata):**
> "O principal feedback da Sprint 1 foi formalizar as User Stories — elas estavam só implícitas nas personas. A gente refez isso direito.
>
> Cada uma das 3 personas agora tem uma US formal com **4 critérios de aceite no formato Dado-Quando-Então**, totalizando 12 critérios rastreáveis aos entregáveis técnicos. Esse aqui é um exemplo do critério do operador pra risco ALTO.
>
> O documento completo tá em `docs/user_stories.md`."

---

## SLIDE 3 — Dataset v2 ampliado (40s)

**No slide:**
- Título: **Dataset v2 — De 5 para 8 features, de ~50 para 1000 linhas**
- Tabela compacta:

| | Sprint 1 | Sprint 2 |
|---|---|---|
| Linhas | ~50 | **1000** |
| Features | 5 | **8** |
| Geração | Aleatória | **Estratificada (45/35/20)** |

- Box destacado das novas variáveis:
  - 🌡️ **temperatura** (°C) — fadiga + superaquecimento
  - 🔧 **idade_maquina** (anos) — falha mecânica
  - ⏰ **horas_operacao** (h) — desgaste + fadiga

**Fala (João):**
> "Outro ponto do feedback foi ampliar o dataset. A gente foi de 5 pra 8 variáveis e de 50 pra 1000 linhas.
>
> As 3 novas variáveis são **temperatura**, **idade da máquina** e **horas de operação acumuladas no dia**. Todas seguem o princípio da Sprint 1: dá pra captar via celular, cadastro ou API — sem depender de telemetria proprietária. Funciona em frota antiga.
>
> E a geração agora é **estratificada em 3 cenários** — seguro, limite e crítico — pra garantir que o modelo veja exemplos suficientes de cada classe."

---

## SLIDE 4 — Modelo treinado + métricas (60s)

**No slide:**
- Título: **Random Forest treinado e validado**
- Esquerda: pipeline visual

```
Features → ColumnTransformer → RandomForest(300 árvores) → classe + score
```

- Direita: métricas em destaque

```
┌─────────────────────────────────────┐
│ Cross-Validation 5-fold             │
│                                     │
│ Acurácia:  90.8% (±1.4%)           │
│ F1 macro:  76.1% (±6.1%)           │
│                                     │
│ Recall ALTO: 90%  ← crítico!        │
└─────────────────────────────────────┘
```

- Embaixo: top 3 features importantes
  - 1. umidade_solo
  - 2. velocidade
  - 3. declividade

**Fala (João):**
> "O modelo é um **Random Forest** com 300 árvores, treinado com pipeline scikit-learn. A escolha foi mantida da Sprint 1 porque tem 4 vantagens: captura interações não-lineares, funciona bem em dataset moderado, lida com features mistas, e é interpretável via feature importance.
>
> Os hiperparâmetros foram ajustados via grid search manual. As métricas finais em **cross-validation 5-fold**: 90.8% de acurácia, com desvio padrão de só 1.4% — o modelo é estável.
>
> O que mais importa pro nosso domínio é o **recall da classe ALTO: 90%**. Falso negativo de ALTO seria deixar passar uma situação realmente perigosa — e o modelo acerta 9 em cada 10."

---

## SLIDE 5 — Demo do dashboard (75s)

**Conteúdo:** Gravação de tela do dashboard rodando ao vivo.

**Roteiro da demo (Renata):**

> *(tela do dashboard com sliders no padrão BAIXO)*
>
> "Esse é o dashboard que a gente montou em Streamlit. Ele consome direto o modelo treinado.
>
> *(começa com cenário seguro)*
>
> Nessa configuração — umidade 50%, declividade 8%, sem chuva — o modelo retorna risco **BAIXO**, score baixo, recomendação 'seguir rota atual'. Nenhuma variável marcada como alta.
>
> *(mexe os sliders pra cenário-limite)*
>
> Agora vou simular uma situação-limite: subindo a umidade pra 65%, declividade pra 14%, e botando 15mm de chuva. Olha o que acontece — o modelo passa pra **MÉDIO**, score sobe, e a recomendação fica contextual: 'reduzir velocidade'.
>
> *(empurra para cenário crítico)*
>
> Empurrando pra crítico: solo saturado em 88%, declividade 22%, chuva pesada de 50mm, temperatura 39°C, máquina velha de 15 anos. Agora ele dispara **ALTO** com 'interromper operação'.
>
> *(rola para baixo até Feature Importance)*
>
> E aqui o ponto-chave pra Sompo: o dashboard mostra **por que** o modelo decidiu. Essa é a feature importance, indicando que umidade do solo, velocidade e declividade foram as variáveis com mais peso. Transparência total — fundamental pra LGPD e pra auditoria de seguro."

---

## SLIDE 6 — Arquitetura consolidada + Roadmap (40s)

**No slide:**
- Esquerda: diagrama de arquitetura atualizado (do README, item 7)
- Direita: o que vem na Sprint 3

```
🆕 Persistência (Oracle XE) explicitada
🆕 Cada saída ligada à US correspondente
🆕 View vw_metricas_diarias alimentando dashboard

Sprint 3:
- Score consolidado do produtor (US-03 CA-03.2)
- Refinamento dos mockups
- Simulação end-to-end completa
```

**Fala (João):**
> "Pra fechar: a arquitetura ganhou uma camada nova — a **persistência no Oracle XE** — com 4 tabelas, uma view de agregação diária e índices pras queries do dashboard. Tudo conectando a saída do modelo às 3 User Stories.
>
> Na Sprint 3 a gente vai trabalhar no **score consolidado do produtor** pra Sompo — que é o critério CA-03.2 que ficou pra última iteração — e refinar a integração end-to-end pro Festival NEXT."

---

## FECHAMENTO (10s)

**No slide:** frase grande + nomes do grupo

> *"Em vez de pagar o sinistro depois, **calculamos o risco antes**."*

**Fala (Renata e João alternando ou em coro):**
> "Esse foi o Sompo AgroPredict — Sprint 2 entregue. **Obrigado!**"

---

## CHECKLIST PRÉ-GRAVAÇÃO

- [ ] Dataset gerado: `dataset_v2.csv` existe
- [ ] Notebook rodado: `modelo.pkl` existe (1.9MB)
- [ ] Dashboard testado: `streamlit run app.py` abre sem erro
- [ ] PowerPoint pronto com os 6 slides
- [ ] Loom ou OBS configurado (gravar tela + microfone + webcam opcional)
- [ ] Ensaiar 2x cronometrando — tem que caber em 5 minutos
- [ ] Apresentadores combinaram quem fala cada slide
- [ ] Para o slide 5 (demo): ter o dashboard rodando ANTES de começar a gravar

## CHECKLIST PÓS-GRAVAÇÃO

- [ ] Upload no YouTube como **não-listado**
- [ ] Copiar o link no README v2 (seção 11)
- [ ] Confirmar que tutora Nicolly tem acesso ao repositório privado
- [ ] Verificar que todos os arquivos da estrutura do README estão commitados
