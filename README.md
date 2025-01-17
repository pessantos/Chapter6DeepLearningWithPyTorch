# Chapter6DeepLearningWithPyTorch

Regressão Linear com PyTorch e Schedulers de Taxa de Aprendizado

Este projeto demonstra a aplicação de otimizadores modernos (Adam e SGD) e schedulers de taxa de aprendizado em um modelo simples de regressão linear, utilizando a biblioteca PyTorch. O objetivo é explorar como diferentes técnicas de otimização e ajuste de taxa de aprendizado podem influenciar o treinamento e a convergência do modelo.

## Descrição do Código

- **Modelo Simples:** Um modelo de regressão linear treinado com dados simulados.
- **Otimizadores:** 
  - `Adam`: Utiliza médias móveis exponencialmente ponderadas para suavizar os gradientes.
  - `SGD`: Inclui suporte para momentum.
- **Schedulers de Taxa de Aprendizado:**
  - `StepLR`: Reduz a taxa de aprendizado após um número fixo de épocas.
  - `CyclicLR`: Alterna a taxa de aprendizado entre valores máximo e mínimo, promovendo exploração cíclica.
- **Visualização:** Exibe os resultados do modelo treinado usando cada otimizador.

## Estrutura do Código

1. **Gerar Dados Simulados:**
   - Geração de dados com ruído para representar uma relação linear.
   - Divisão em conjuntos de treinamento e validação.
   
2. **Definir Modelo:**
   - Modelo de regressão linear implementado com `nn.Linear`.
   
3. **Configurar Otimizadores e Schedulers:**
   - Configuração de `Adam` e `SGD` com diferentes schedulers para controlar a taxa de aprendizado.
   
4. **Treinamento:**
   - O modelo é treinado com cada otimizador e scheduler.
   - Perdas de treinamento e validação são monitoradas a cada época.

5. **Visualização:**
   - Resultados são plotados para comparar o desempenho de ambos os métodos.

## Como Executar

1. **Pré-requisitos:**
   - Python 3.8 ou superior.
   - Bibliotecas necessárias:
     ```bash
     pip install torch matplotlib numpy
     ```

2. **Executar o Script:**
   - Salve o código em um arquivo chamado `training_with_schedulers.py`.
   - Execute o script no terminal:
     ```bash
     python training_with_schedulers.py
     ```

3. **Resultados:**
   - O script exibe as perdas de treinamento e validação no terminal.
   - Um gráfico compara os resultados do modelo treinado com `Adam` e `SGD`.

## Resultados Esperados

- **Adam:** Convergência mais estável devido às médias móveis adaptativas.
- **SGD:** Exploração cíclica de taxas de aprendizado com o `CyclicLR`.
- **Visualização:** Gráfico comparativo mostrando os modelos treinados.

## Personalizações

- **Parâmetros de Treinamento:** Ajuste as taxas de aprendizado, número de épocas e parâmetros dos schedulers para experimentar diferentes configurações.
- **Dados:** Modifique o conjunto de dados para explorar diferentes relações.
