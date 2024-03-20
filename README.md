# Pytorch_Lightning
PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.
  ![image](https://github.com/WeberSouzaWeb/Pytorch_Lightning/assets/107212929/507113fe-6aa4-4923-90c6-b93f0fbfae8e)
## Carregando os Dados e Definindo as Transformações
  Para este projeto o objetivo é construir um modelo de classificação de imagens, ou seja, um modelo de Visão Computacional. Dada uma imagem o modelo deve entregar a probabilidade de classe a qual a imagem pertence.

  O conjunto de dados contém imagens de dígitos de 0 a 9. Cada imagem tem um dígito em diferentes formatos. Logo, cada imagem pode pertencer a uma de 10 classes possíveis. Criaremos um modelo de classificação multiclasse.
  Aplica-se uma edição personalizada nas imagens, para que todas as imagens tenham as mesmas dimensões e propriedades. Fazemos isso usando torchvision.transforms.
  - O método transforms.ToTensor() converte a imagem em números, que são compreensíveis pelo sistema. Ele separa a imagem em três canais de cores: vermelho, verde e azul (RGB). Em seguida, converte os pixels de cada imagem no brilho de suas cores entre 0 a 255. Esses valores são reduzidos para um intervalo entre 0 e 1. A imagem agora é im tensor PyTorch.
  - O método transforms.Normalize() normaliza o tensor colocando os valores na mesma escala.

  O formato das imagens torch.Size([64,1,28,28]), indica que existem 64 imagens em cada lote e cada imagem tem uma dimensão de 28 x 28 pixels. Da mesma forma, os rótulos têm a forma de torch.Size([64]).
## O que são Funções de Ativação?
  As funções de ativação são usadas em redes neurais para introduzir não linearidade em suas camadas.

  Em redes neurais profundas, as funções de ativação permitem que os modelos aprendam padrões complexos e abstrair informações úteis a partir do dados de entrada. Algumas das funções de ativação mais comuns incluem a ReLU (unidade linear retificada), a sigmóide e a tangente hiperbólica.

  A ReLU é uma função de ativação bastante popular, pois tem um bom desempenho e é fácil de implementar. Ela é definida como f(x) = max(0,x), ou seja, é simplesmente o valor máximo entre zero e o valor de entrada x.

  Sigmóide é outra função de ativação comum, que é útil para problemas de classificação binária. Ela é definida como f(x) = 1/(1+e^(-x)).

  A tangente hiperbólica é outra função de ativação que é frenquentemente usada em redes neurais. Ela é definida como f(x) = tanh(x)

  Essas são apenas algumas das muitas funções de ativação disponíveis e casa uma delas pode ser mais adequada para determinados tipos de problemas. É importante experimentar diferentes funções de ativação para ver qual tem o melhor desempenho para o seu problema específico.

  No link abaixo você encontrará uma lista constantemente atualizada de funções de ativação com o ano de lançamento de cada uma, além das principais aplicações e usos:
  https://paperswithcode.com/methods/category/activation-functions

## Função de Ativação ReLU

  ReLU significa função de ativação linear retificada. É uma função não linear e, graficamente, ReLU tem o seguinte comportamento transformativo:
  ![image](https://github.com/WeberSouzaWeb/Pytorch_Lightning/assets/107212929/f0dc03f8-eada-4e2c-a12d-e310c34a14e8)
  ReLU é uma função de ativação popular, pois é diferenciável e não linear. Se as entradas forem negativas, sua derivada torna-se zero, o que causa a "morte" dos neurônios e o aprendizado não ocorre. A ReLU permite a passagem de valores positivos, enquanto valores negativos sao modificados para zero.

## Função de Ativação LeakyReLU

  Leaky ReLU Activation Function ou LReLU é outro tipo de funçãode ativação que é semelhantea ReLU, mas resolve o problema de neurônios 'mortos' e, graficamente, a Leaky ReLU tem o seguinte comportamento transformativo:
  ![image](https://github.com/WeberSouzaWeb/Pytorch_Lightning/assets/107212929/9a86b91d-4054-4c62-9a31-b1480cf65adb)
  Esta função é muito útil porque quando a entrada é negativa a diferenciação da função não é zero. Portanto, o aprendizado dos neurônios não para.

