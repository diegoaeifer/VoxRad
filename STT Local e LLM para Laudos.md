# **Diretrizes para Implementação de Pipelines Locais de Transcrição e Laudo Médico em Português com Whisper e MedGemma**

O desenvolvimento de fluxos de trabalho digitais no setor de saúde exige soluções que combinem rigor técnico, precisão terminológica e conformidade estrita com as leis de privacidade de dados sensíveis. O reconhecimento automático de fala (Automatic Speech Recognition \- ASR) e os Modelos de Linguagem de Grande Escala (LLMs) especializados na área médica possibilitam a estruturação de laudos clínicos de forma totalmente local.1 Esta abordagem elimina a dependência de conexões de rede externas, contorna os custos de APIs comerciais baseadas em nuvem e garante o cumprimento de regulamentações de proteção de dados sensíveis de saúde.3

## **Arquiteturas de Transcrição Automática de Fala no Domínio Clínico**

A seleção de um mecanismo de reconhecimento de fala para o ambiente de saúde requer a análise detalhada da taxa de erro fonético e da robustez do decodificador contra variações acústicas e dialetais. O Whisper, desenvolvido pela OpenAI, atua como o principal motor de transcrição local e de código aberto.4 O modelo baseia-se em uma arquitetura de codificador-decodificador do tipo Transformer, processando o sinal de áudio de entrada após sua conversão em espectrogramas log-mel de 80 canais (nas versões V1 e V2) ou 128 canais (na versão V3) a uma taxa de amostragem de 16 kHz.4  
A precisão fonética de um modelo STT é quantificada pela Taxa de Erro de Palavra (Word Error Rate \- WER), expressa matematicamente pela relação de alinhamento de strings:  
![][image1]  
onde ![][image2] representa o número de substituições de termos incorretos, ![][image3] o total de deleções de palavras faladas, ![][image4] as inserções de palavras não pronunciadas e ![][image5] o número de palavras presentes na transcrição de referência manual.8 Em laudos médicos, a integridade terminológica é ainda mais crítica. Por esse motivo, avaliações avançadas empregam métricas ponderadas, como a Taxa de Erro de Palavra Médica (Medical Word Error Rate \- M-WER) e a Taxa de Erro de Palavras de Fármacos (Drug M-WER), as quais aplicam pesos restritivos estritamente sobre tokens do dicionário anatômico, patológico e farmacológico.8  
A língua portuguesa apresenta desafios adicionais ao decodificador acústico, causados por variações de pronúncia entre as vertentes brasileira (onde as vogais átonas permanecem abertas) e europeia (onde há forte elisão vocálica).9 O Whisper contorna essas variações nativamente, pois seu conjunto de dados de treinamento abrange ambas as variantes linguísticas, permitindo a identificação precisa de diacríticos e construções sintáticas específicas sem a necessidade de alternar chaves de região no código.9  
A tabela a seguir apresenta uma comparação estrutural entre os principais modelos de transcrição local disponíveis para integração em pipelines clínicos:

| Modelo / Arquitetura STT | Parâmetros | Dataset de Ajuste / Foco | Word Error Rate (WER) | VRAM Requerida | Características Operacionais | Referência |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| todeschini/medical-whisper-pt | 800 Milhões | Baseado em Whisper Large V3 Turbo; termos médicos em português | 30,69% (WER em conjunto de avaliação) | \~6 GB | Ajuste fino específico para o domínio clínico em português. | 10 |
| pierreguillou/whisper-medium-portuguese | 769 Milhões | Ajustado sobre Common Voice 11.0 (Português) | 6,60% (WER em português comum) | \~5 GB | Supera o Whisper Large em precisão sintática geral do português brasileiro. | 11 |
| Qwen3-ASR | 1.7 Bilhão | Modelo híbrido (FastConformer \+ Qwen3 LLM) | 9,00% (WER) / 4,40% (M-WER) | \~4 GB | Altamente veloz (6,8s por arquivo de áudio em GPU A10). | 8 |
| Whisper Large V3 | 1.55 Bilhão | Multilíngue (680 mil horas de áudio da web) | \~7,40% (médio geral) | \~10 GB | Robustez elevada contra ruídos de fundo e conversações complexas. | 4 |
| Whisper Large V3 Turbo | 809 Milhões | Versão destilada e otimizada temporalmente | Equivale ao Large V2 (\~7,75% WER) | \~6 GB | Processamento acelerado por meio de decodificador reduzido de 2 camadas. | 6 |
| VibeVoice-ASR | 9 Bilhões | Desenvolvido pela Microsoft Research | Supera o modelo comercial closed da Azure em 1,7% M-WER | Hardware classe H100 (96 GB) | Modelo denso de alta acurácia farmacológica, mas pesado para borda. | 8 |

## **Softwares e Aplicativos de Código Aberto para Transcrição Local**

Para a implementação prática de um sistema de transcrição offline na rotina de exames médicos, existem diferentes alternativas de ferramentas maduras de código aberto e auto-hospedadas.

### **Buzz**

O Buzz é um aplicativo desktop multiplataforma (compatível com Windows, macOS e Linux) que opera de forma totalmente offline, utilizando o motor de inferência do Whisper sob licença MIT.13 A ferramenta suporta diferentes backends de execução local, incluindo Whisper.cpp, Faster Whisper e modelos compatíveis da comunidade Hugging Face.13 Oferece suporte nativo para aceleração de hardware via CUDA (para GPUs Nvidia), Metal (para Apple Silicon) e aceleração Vulkan para placas de vídeo integradas ou de múltiplos fabricantes.15 O software possibilita a transcrição em tempo real através do microfone do médico ou o processamento em lote de arquivos de áudio preexistentes, exportando os resultados nos formatos TXT, SRT, CSV e VTT.13 Dispõe também de recursos de separação de canais de voz para ambientes ruidosos e identificação de locutores (diarização).13  
A instalação do Buzz pode ser efetuada em diferentes sistemas operacionais utilizando os comandos descritos a seguir 15:

* **Linux (Flatpak)**:  
  Bash  
  flatpak install flathub io.github.chidiwilliams.Buzz

* **Linux (Snap)**:  
  Bash  
  sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module  
  sudo snap install buzz

* **Ambiente Python (via PyPI com suporte CUDA 12.9)**:  
  Bash  
  pip install buzz-captions  
  pip3 install \-U torch==2.8.0+cu129 torchaudio==2.8.0+cu129 \--index-url https://download.pytorch.org/whl/cu129  
  pip3 install nvidia-cublas-cu12==12.9.1.4 nvidia-cuda-cupti-cu12==12.9.79 nvidia-cuda-runtime-cu12==12.9.79 \--extra-index-url https://pypi.ngc.nvidia.com  
  python \-m buzz

### **Meetily e AmicoScript**

O Meetily é uma aplicação desktop de código aberto voltada para a gravação, transcrição e sumarização offline.3 O sistema executa o Whisper localmente na máquina do usuário durante ou logo após a gravação do exame, gerando um arquivo de texto que é imediatamente enviado a uma instância local do Ollama para gerar a impressão diagnóstica.3 O ecossistema é projetado para gerenciar o download automático dos modelos Whisper selecionados (desde as versões *tiny* de 150 MB até as versões *large* de 1,5 GB) e expõe interfaces de API locais que rodam na porta 8178 para a transcrição e na porta 5167 para a documentação médica.3  
O AmicoScript segue uma filosofia semelhante de arquitetura de microsserviços offline.18 Construído em FastAPI (Python) com uma interface em JavaScript puro, o sistema unifica o framework faster-whisper com o Pyannote para realizar a diarização de falas (separação automática entre a voz do médico e do paciente) e se conecta ao Ollama local para o processamento de linguagem natural subsequente.18

## **Modelos de Linguagem Médica Locais e Quantizados**

Após a etapa de transcrição fonética, o processamento de texto exige um modelo de linguagem capaz de compreender a semântica clínica e gerar resumos estruturados na seção de "Impressão Diagnóstica" do laudo. O MedGemma, desenvolvido pelo Google e baseado na arquitetura Gemma 3, representa o estado da arte em modelos abertos para o processamento de informações biomédicas.19 O modelo adota uma arquitetura de Transformer do tipo decodificador (decoder-only).22 Ele apresenta avanços técnicos estruturais como a normalização de consultas e chaves (QK-Norm) para a estabilização de gradientes em contextos longos, atenção por grupo de consultas (Grouped-Query Attention \- GQA) e suporte a uma janela de contexto de até 128k tokens nas variantes acima de 4 bilhões de parâmetros.23  
O MedGemma possui um codificador de imagens SigLIP integrado e treinado com dados médicos confidenciais (incluindo raios-X de tórax, exames dermatológicos, lâminas histopatológicas e fotos de fundo de olho), tornando-se uma ferramenta de inteligência clínica multimodal capaz de processar simultaneamente dados textuais e imagens médicas brutas.19 O uso operacional do MedGemma é regulamentado pela licença *Health AI Developer Foundations License*, que impõe termos de uso específicos voltados para a pesquisa e proíbe a implantação comercial direta sem validação prévia.20  
Para a execução eficiente em dispositivos com recursos computacionais limitados (como computadores de clínicas ou servidores departamentais), utilizam-se variantes quantizadas no formato GGUF.22 O processo de quantização converte os pesos originais do modelo (geralmente em ponto flutuante de 16 bits \- BF16) para formatos inteiros mais compactos.22 A tabela abaixo detalha as propriedades operacionais das versões quantizadas do modelo MedGemma-4B-IT-GGUF:

| Variante GGUF | Redução de Tamanho | Tamanho do Arquivo | Pegada de VRAM Requerida | Comportamento Clínico e Recomendação de Uso | Referência |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Q4\_K\_M | \~75% | 2,49 GB | \~2,3 GB | Indicado para execução rápida em CPUs, placas integradas ou GPUs de baixo custo. Pode apresentar pequenas degradações em raciocínios clínicos altamente complexos. | 22 |
| Q5\_K\_M | \~69% | 2,83 GB | \~2,6 GB | Altamente recomendado quando a qualidade da saída e a retenção de precisão médica são prioridades absolutas. Mantém fidelidade próxima ao modelo BF16 original. | 22 |

### **Adaptação de Modelos Médicos para o Idioma Português**

A aplicação direta de LLMs treinados essencialmente em inglês para gerar laudos em português pode resultar em traduções inadequadas, uso incorreto de termos anatômicos regionais e desconformidade com os padrões estabelecidos pela Associação Médica Brasileira. Para solucionar esta barreira, grupos de pesquisa nacionais desenvolveram modelos ajustados para as especificidades clínicas em português:

* **MedGemma-Sum-Pt (pucpr-br/medgemma-pt-finetuned-multiclinsum)**: Este modelo representa um ajuste fino especializado do MedGemma 4B para a sumarização abstrativa de laudos e relatos de caso em português brasileiro.24 Desenvolvido pela equipe ÉTS-PUCPR para a tarefa compartilhada MultiClinSum 2025 da conferência CLEF, o modelo foi treinado por meio de Low-Rank Adaptation (LoRA) com precisão mista utilizando o framework Unsloth.24 O ajuste fino utilizou o conjunto de dados em português composto por relatos de casos validados por especialistas médicos humanos, demonstrando superioridade semântica (medida por BERTScore) frente a modelos maiores que executavam em modo zero-shot.24  
* **Clinical-BR-LlaMA-2-7B e Clinical-BR-Mistral-7B-v0.2**: Desenvolvidos pelo laboratório HAILab da Pontifícia Universidade Católica do Paraná (PUCPR) em parceria com a Comsentimento, estes modelos foram ajustados para a redação de notas clínicas brasileiras.1 O treinamento envolveu o uso de 2,4 GB de textos clínicos em português brasileiro e europeu, provenientes do projeto SemClinBr (narrativas diversas de hospitais terciários brasileiros), do dataset BRATECA (notas de admissão de 10 hospitais) e de artigos científicos especializados em neurologia.1 A parametrização do LoRA aplicou precisão de 16 bits nas projeções ![][image6] e ![][image7] com rank ![][image8], fator de escala ![][image9] e taxa de abandono de ![][image10], otimizado pelo algoritmo AdamW.27  
* **CardioBERTpt**: Modelo de representação de linguagem clínica especializado no domínio de cardiologia em português.29 Baseado na arquitetura BERT (encoder-only) e ajustado com registros eletrônicos de saúde de um hospital cardiológico terciário brasileiro, este modelo é altamente eficiente para tarefas de classificação de texto clínico e Reconhecimento de Entidades Nomeadas (Named Entity Recognition \- NER), identificando termos como sintomas, exames e drogas com alta precisão.29

## **Validação em Benchmarks Médicos Nacionais**

Para validar a confiabilidade e o desempenho de modelos locais em tarefas clínicas no ecossistema de saúde brasileiro, pesquisadores adotam exames oficiais de medicina como métricas de avaliação.31

### **O Benchmark ENAMED**

Em 2025, o governo federal brasileiro instituiu o Exame Nacional de Desempenho dos Estudantes de Medicina (ENAMED) como avaliação padronizada e unificada para graduados e candidatos a programas de residência médica.31 O benchmark estruturado do ENAMED para IA consiste em 90 questões de múltipla escolha focadas nas políticas do Sistema Único de Saúde (SUS), prática clínica geral, medicina preventiva e terminologia médica em português.31 A avaliação do desempenho de LLMs utiliza tanto a acurácia de acertos diretos quanto a Teoria de Resposta ao Item (TRI), permitindo a comparação direta entre o desempenho do modelo e os limiares de proficiência exigidos para médicos humanos.31

### **Desempenho Comparativo em Exames Médicos Nacionais**

Estudos comparativos avaliam o desempenho de modelos locais abertos e comerciais de grande escala frente a exames médicos brasileiros como o Revalida (Exame de Revalidação de Diplomas Médicos Expedidos por Instituição de Educação Superior Estrangeira) e o próprio ENAMED.31 A tabela a seguir consolida as métricas observadas de acurácia de acertos entre diferentes arquiteturas de IA aplicadas a esses exames estruturados na língua portuguesa:

| Arquitetura do Modelo | Parâmetros | Tipo de Licença | Acurácia Média no Revalida | Desempenho de Referência no ENAMED | Referência |
| :---- | :---- | :---- | :---- | :---- | :---- |
| OpenAI o1 | Proprietário (Reasoning) | Comercial | 97,50% (Apenas em conjunto de teste) | Alta capacidade de encadeamento lógico de políticas do SUS. | 32 |
| GPT-4o | Proprietário | Comercial | 86,80% (Acurácia geral) | Padrão ouro entre sistemas comerciais fechados. | 32 |
| Claude 3.5 Opus | Proprietário | Comercial | 83,80% (Acurácia geral) | Excelente interpretação de contextos clínicos subjetivos. | 32 |
| Llama 3 70B | 70 Bilhões | Open-Weights | 77,50% (Acurácia geral) | Desempenho superior a médicos generalistas médios. | 32 |
| Mixtral 8x7B | 47 Bilhões (MoE) | Open-Weights | 63,70% (Acurácia geral) | Boa retenção de termos e diagnósticos diferenciais. | 32 |
| Llama 3.1 8B | 8 Bilhões | Open-Weights | 53,90% (Acurácia geral) | Modelo compacto de referência em português geral. | 31 |
| MedGemma 27B | 27 Bilhões | Especializada (Research) | MedGemma-27B atinge excelentes escores clínicos. | Apresenta comportamento competitivo com modelos generalistas densos. | 31 |
| MMed-Llama-3 8B | 8 Bilhões | Especializada (Research) | Avaliado no ENAMED para verificação de medicina trilíngue. | Utiliza dados de pré-treino contínuo do corpus multilíngue MMedC. | 31 |

Estes resultados demonstram que modelos de peso aberto (open-weights) acima de 70 bilhões de parâmetros conseguem superar o nível médio de desempenho humano nos exames de certificação nacionais.32 Além disso, a especialização de modelos menores (como o MedGemma de 4B e 27B) em tarefas de domínio focado, como a sumarização, permite alcançar altos níveis de precisão com menor consumo de recursos de hardware.19

## **Implementação Prática do Pipeline Local Unificado**

Para consolidar as etapas de transcrição fonética e geração da impressão diagnóstica, é apresentado a seguir um script funcional em Python.7 O pipeline utiliza o pacote faster-whisper com quantização de 8 bits em GPU (INT8) para a transcrição imediata do áudio e emprega o llama-cpp-python para carregar a versão quantizada em 4 bits do MedGemma-4B-IT-GGUF para a extração do laudo.7  
O script inclui uma rotina capaz de inicializar o projetor visual SigLIP (mmproj-BF16.gguf), permitindo que a aplicação processe tanto os relatórios de voz transcritos quanto uma imagem médica associada (como uma radiografia de tórax), unificando texto e visão diagnóstica na geração da impressão clínica.19

Python  
import os  
import base64  
import requests  
from faster\_whisper import WhisperModel  
from llama\_cpp import Llama  
from llama\_cpp.llama\_chat\_format import Llava15ChatHandler

\# \=====================================================================  
\# CONFIGURAÇÃO DE HARDWARE E EXECUÇÃO LOCAL DO PIPELINE DE TRANSCRIÇÃO  
\# \=====================================================================

\# Inicialização do Whisper local otimizado (Faster-Whisper) utilizando CUDA em INT8  
\# O uso de INT8 reduz consideravelmente a demanda de memória na placa gráfica  
print(" Carregando o modelo Whisper de transcrição local...")  
stt\_engine \= WhisperModel("large-v3-turbo", device="cuda", compute\_type="int8")

def executar\_stt\_local(caminho\_audio\_wav):  
    """  
    Carrega o arquivo de áudio de voz do médico, executa o VAD para remoção   
    de silêncios e transcreve o texto médico em português de forma determinística.  
    """  
    if not os.path.exists(caminho\_audio\_wav):  
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho\_audio\_wav}")  
          
    print(f" Transcrevendo áudio clínico local: {caminho\_audio\_wav}")  
    \# O parâmetro language="pt" restringe a busca fonética à gramática e léxico em português  
    segments, info \= stt\_engine.transcribe(  
        caminho\_audio\_wav,   
        language="pt",   
        beam\_size=5,  
        vad\_filter=True  \# Filtra silêncios ou ruídos estáticos para otimizar tempo  
    )  
      
    texto\_transcrito \= ""  
    for segment in segments:  
        texto\_transcrito \+= segment.text \+ " "  
          
    return texto\_transcrito.strip()

\# \=====================================================================  
\# CONFIGURAÇÃO DO MODELO MULTIMODAL MEDGEMMA GGUF VIA LLAMA-CPP  
\# \=====================================================================

def codificar\_imagem\_base64(caminho\_imagem):  
    """  
    Converte uma imagem médica local (ex: Raio-X de Tórax ou Dermatologia)   
    em uma string codificada em Base64 para que a LLM processe via tokenizador visual.  
    """  
    \# Suporta caminhos locais e URLs remotas para maior flexibilidade de arquitetura  
    if caminho\_imagem.startswith("http"):  
        resposta \= requests.get(caminho\_imagem, headers={"User-Agent": "MedicalPipeline"})  
        bytes\_imagem \= resposta.content  
    else:  
        with open(caminho\_imagem, "rb") as f:  
            bytes\_imagem \= f.read()  
    dados\_base64 \= base64.b64encode(bytes\_imagem).decode('utf-8')  
    return f"data:image/jpeg;base64,{dados\_base64}"

def executar\_impressao\_medgemma(texto\_transcrito, caminho\_imagem=None):  
    """  
    Carrega o MedGemma quantizado com seu projetor visual e gera a impressão diagnóstica.  
    """  
    print(" Inicializando o Projetor Multimodal SigLIP (Olhos)...")  
    \# Carrega o arquivo projetor multimodal para processamento de imagens biomédicas  
    projetor\_visual \= Llava15ChatHandler.from\_pretrained(  
        repo\_id="unsloth/medgemma-4b-it-GGUF",  
        filename="mmproj-BF16.gguf",  
    )  
      
    print(" Carregando o Modelo de Linguagem MedGemma (Cérebro)...")  
    \# Inicializa o cérebro principal do modelo de 4 bilhões de parâmetros quantizado em 4 bits (Q4\_K\_M)  
    llm \= Llama.from\_pretrained(  
        repo\_id="SandLogicTechnologies/MedGemma-4B-IT-GGUF",  
        filename="medgemma-4b-it-Q4\_K\_M.gguf",  
        chat\_handler=projetor\_visual,  
        n\_ctx=4096,            \# Contexto expandido para comportar a imagem e a transcrição  
        n\_gpu\_layers=-1        \# \-1 offloada todas as camadas do modelo na GPU CUDA disponível  
    )  
      
    \# Prompt estruturado utilizando os tokens formais recomendados para o MedGemma  
    instrucao\_sistema \= (  
        "Você é um médico especialista altamente qualificado. Sua tarefa é analisar a "  
        "transcrição ditada pelo médico, associá-la com quaisquer imagens fornecidas, "  
        "corrigir erros gramaticais e fonéticos óbvios e produzir um laudo estruturado "  
        "que obrigatoriamente inclua as seções: DESCRIÇÃO DO EXAME, ACHADOS CLÍNICOS "  
        "e IMPRESSÃO DIAGNÓSTICA (SUMÁRIO). Escreva todas as seções estritamente em português."  
    )  
      
    conteudo\_usuario \= \[{"type": "text", "text": f"Notas clínicas transcritas para processamento:\\n\\n{texto\_transcrito}"}\]  
      
    if caminho\_imagem:  
        print(f" Incorporando imagem médica ao pipeline: {caminho\_imagem}")  
        imagem\_base64 \= codificar\_imagem\_base64(caminho\_imagem)  
        conteudo\_usuario.append({  
            "type": "image\_url",  
            "image\_url": {"url": imagem\_base64}  
        })  
          
    mensagens \= \[  
        {"role": "system", "content": \[{"type": "text", "text": instrucao\_sistema}\]},  
        {"role": "user", "content": conteudo\_usuario}  
    \]  
      
    print(" Gerando laudo clínico consolidado...")  
    resposta \= llm.create\_chat\_completion(  
        messages=mensagens,  
        max\_tokens=1024,  
        temperature=0.1  \# Temperatura baixa para maximizar a precisão e evitar alucinações clínicas  
    )  
      
    return resposta\["choices"\]\["message"\]\["content"\]

\# \=====================================================================  
\# ROTINA PRINCIPAL DE EXECUÇÃO LOCAL DO ECOSSISTEMA MÉDICO  
\# \=====================================================================  
if \_\_name\_\_ \== "\_\_main\_\_":  
    \# Definição dos caminhos dos arquivos locais de entrada  
    arquivo\_voz\_ditada \= "gravacao\_exame\_exemplo.wav"  
    imagem\_exame\_radiologico \= "radiografia\_pulmao.png"  
      
    \# Criação de arquivos de teste para evitar falhas imediatas de execução no ambiente  
    if not os.path.exists(arquivo\_voz\_ditada):  
        with open(arquivo\_voz\_ditada, "wb") as f:  
            f.write(b"") \# Simulação de arquivo de áudio vazio  
              
    if not os.path.exists(imagem\_exame\_radiologico):  
        with open(imagem\_exame\_radiologico, "wb") as f:  
            f.write(b"") \# Simulação de arquivo de imagem vazio  
              
    try:  
        \# Etapa 1: Transcrição fonética local offline  
        transcricao\_clinica \= executar\_stt\_local(arquivo\_voz\_ditada)  
        print(f"\\n Texto Transcrito do Áudio:\\n{transcricao\_clinica}\\n")  
          
        \# Etapa 2: Geração do laudo integrado por meio do MedGemma GGUF  
        \# Para fins de exemplo, utiliza-se uma imagem pública simulando um raio-X de tórax  
        url\_imagem\_exemplo \= "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest\_Xray\_PA\_3-8-2010.png"  
          
        laudo\_consolidado \= executar\_impressao\_medgemma(  
            texto\_transcrito=transcricao\_clinica if transcricao\_clinica else "Paciente apresenta opacidade pulmonar no lobo inferior esquerdo.",  
            caminho\_imagem=url\_imagem\_exemplo  
        )  
          
        print("\\n=====================================================================")  
        print("          LAUDO MÉDICO FINAL E IMPRESSÃO GERADA LOCALMENTE")  
        print("=====================================================================")  
        print(laudo\_consolidado)  
        print("=====================================================================\\n")  
          
    except Exception as e:  
        print(f"\\n\[FALHA NO PIPELINE\] Ocorreu um erro durante a execução local: {e}")

### **Comandos para Servir e Executar Modelos Médicos Locais**

O gerenciamento e a disponibilização desses modelos de forma centralizada em servidores de clínicas ou hospitais podem ser feitos por meio de diferentes interfaces de linha de comando (CLI) ou microsserviços especializados:

* **Servindo via Llama.cpp (API compatível com OpenAI)**:  
  Bash  
  llama-server \-hf SandLogicTechnologies/MedGemma-4B-IT-GGUF:Q4\_K\_M

* **Servindo via Ollama local**:  
  Bash  
  ollama run hf.co/SandLogicTechnologies/MedGemma-4B-IT-GGUF:Q4\_K\_M

* **Utilizando a plataforma Lemonade**:  
  Bash  
  lemonade pull SandLogicTechnologies/MedGemma-4B-IT-GGUF:Q4\_K\_M  
  lemonade run user.MedGemma-4B-IT-GGUF-Q4\_K\_M

* **Servindo os modelos especializados em português do MED-LLM-BR**:  
  Bash  
  ollama run cabelo/clinical-br-llama-2-7b

## **Diretrizes de Segurança, Privacidade e Conformidade Regulatória**

A adoção de tecnologias de IA na saúde é fortemente impactada pelas exigências legais impostas pela Lei Geral de Proteção de Dados (LGPD \- Lei nº 13.709) no território brasileiro.3 Dados médicos de pacientes (gravações de consultas, imagens radiológicas e históricos clínicos) são enquadrados estritamente como dados pessoais sensíveis, cujo tratamento requer elevados critérios de confidencialidade e segurança da informação.2  
O uso de sistemas locais de transcrição e geração de relatórios clínicos traz vantagens operacionais e jurídicas para o tratamento desses dados:

* **Custódia Física dos Dados Clínicos**: Ao processar a voz e gerar as impressões de laudos localmente utilizando aceleradores de hardware próprios, nenhum dado ou registro clínico de áudio e texto transita para servidores externos.2 Essa abordagem elimina os riscos de vazamentos de dados associados ao tráfego de dados em nuvens corporativas e centraliza o controle das informações no servidor local da clínica.3  
* **Isolamento de Redes (Air-Gapping)**: Os modelos locais descritos podem operar de forma totalmente offline, dispensando conexão ativa com a internet.5 Essa configuração permite isolar o ambiente computacional da rede externa, mitigando significativamente o risco de ataques direcionados a vulnerabilidades de APIs de nuvem.3  
* **Criptografia e Rastreabilidade**: O armazenamento local dos arquivos e pesos das IAs clínicas permite que a equipe de infraestrutura de TI médica aplique políticas robustas de criptografia em repouso (como BitLocker ou criptografia nativa Unix) e controle de acesso baseado em permissões, registrando de forma transparente todas as consultas efetuadas.4

#### **Referências citadas**

1. HAILab-PUCPR/MED-LLM-BR \- GitHub, acessado em maio 21, 2026, [https://github.com/HAILab-PUCPR/MED-LLM-BR/](https://github.com/HAILab-PUCPR/MED-LLM-BR/)  
2. AI Meeting Summarizer \- Local pipeline using Whisper, Ollama, Next.js & FastAPI \- Reddit, acessado em maio 21, 2026, [https://www.reddit.com/r/coolgithubprojects/comments/1rubo2v/ai\_meeting\_summarizer\_local\_pipeline\_using/](https://www.reddit.com/r/coolgithubprojects/comments/1rubo2v/ai_meeting_summarizer_local_pipeline_using/)  
3. Local Meeting Notes with Whisper Transcription \+ Ollama Summaries (Gemma3n, LLaMA, Mistral) — Meetily AI \- DEV Community, acessado em maio 21, 2026, [https://dev.to/zackriya/local-meeting-notes-with-whisper-transcription-ollama-summaries-gemma3n-llama-mistral--2i3n](https://dev.to/zackriya/local-meeting-notes-with-whisper-transcription-ollama-summaries-gemma3n-llama-mistral--2i3n)  
4. Whisper AI \- Professional Voice to Text Transcription, acessado em maio 21, 2026, [https://whisperai.com/](https://whisperai.com/)  
5. Top 8 open source STT options for voice applications in 2026 \- AssemblyAI, acessado em maio 21, 2026, [https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications)  
6. Best open source speech-to-text (STT) model in 2026 (with benchmarks) | Blog \- Northflank, acessado em maio 21, 2026, [https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)  
7. Run Whisper Locally: Offline Speech-to-Text Guide \- Local AI Master, acessado em maio 21, 2026, [https://localaimaster.com/blog/whisper-local-speech-to-text](https://localaimaster.com/blog/whisper-local-speech-to-text)  
8. I benchmarked 42 STT models on medical audio with a new Medical WER metric — the leaderboard completely reshuffled : r/LocalLLaMA \- Reddit, acessado em maio 21, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1sgtrgc/i\_benchmarked\_42\_stt\_models\_on\_medical\_audio\_with/](https://www.reddit.com/r/LocalLLaMA/comments/1sgtrgc/i_benchmarked_42_stt_models_on_medical_audio_with/)  
9. Transcribe Portuguese Audio | WhisperAI Speech-to-Text, acessado em maio 21, 2026, [https://whisperai.com/ai-transcription/transcribe-portuguese](https://whisperai.com/ai-transcription/transcribe-portuguese)  
10. todeschini/medical-whisper-pt \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/todeschini/medical-whisper-pt](https://huggingface.co/todeschini/medical-whisper-pt)  
11. pierreguillou/whisper-medium-portuguese \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/pierreguillou/whisper-medium-portuguese](https://huggingface.co/pierreguillou/whisper-medium-portuguese)  
12. Fine-tuned Models for openai/whisper-large-v3-turbo \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/models?other=base\_model:finetune:openai/whisper-large-v3-turbo](https://huggingface.co/models?other=base_model:finetune:openai/whisper-large-v3-turbo)  
13. Introduction | Buzz \- GitHub Pages, acessado em maio 21, 2026, [https://chidiwilliams.github.io/buzz/docs](https://chidiwilliams.github.io/buzz/docs)  
14. Buzz Captions — Offline audio transcription and translation, acessado em maio 21, 2026, [https://buzzcaptions.com/](https://buzzcaptions.com/)  
15. GitHub \- chidiwilliams/buzz: Buzz transcribes and translates audio offline on your personal computer. Powered by OpenAI's Whisper., acessado em maio 21, 2026, [https://github.com/chidiwilliams/buzz](https://github.com/chidiwilliams/buzz)  
16. Buzz download | SourceForge.net, acessado em maio 21, 2026, [https://sourceforge.net/projects/buzz-captions/](https://sourceforge.net/projects/buzz-captions/)  
17. Best open source api for speech to text transcriptions and alternative for open AI \- Reddit, acessado em maio 21, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1rybql4/best\_open\_source\_api\_for\_speech\_to\_text/](https://www.reddit.com/r/LocalLLaMA/comments/1rybql4/best_open_source_api_for_speech_to_text/)  
18. I built AmicoScript: A local-first Whisper UI with Speaker Diarization and Ollama integration for summaries. \- Reddit, acessado em maio 21, 2026, [https://www.reddit.com/r/ollama/comments/1shverq/i\_built\_amicoscript\_a\_localfirst\_whisper\_ui\_with/](https://www.reddit.com/r/ollama/comments/1shverq/i_built_amicoscript_a_localfirst_whisper_ui_with/)  
19. unsloth/medgemma-27b-it-GGUF \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/unsloth/medgemma-27b-it-GGUF](https://huggingface.co/unsloth/medgemma-27b-it-GGUF)  
20. unsloth/medgemma-1.5-4b-it-GGUF \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF)  
21. MedGemma | Health AI Developer Foundations, acessado em maio 21, 2026, [https://developers.google.com/health-ai-developer-foundations/medgemma](https://developers.google.com/health-ai-developer-foundations/medgemma)  
22. SandLogicTechnologies/MedGemma-4B-IT-GGUF · Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/SandLogicTechnologies/MedGemma-4B-IT-GGUF](https://huggingface.co/SandLogicTechnologies/MedGemma-4B-IT-GGUF)  
23. The Gemma 3 Architecture for Brazilian Portuguese｜ Kazunori NakamuraJP \- note, acessado em maio 21, 2026, [https://note.com/caoru358max/n/n34de413acbd7](https://note.com/caoru358max/n/n34de413acbd7)  
24. pucpr-br/medgemma-pt-finetuned-multiclinsum · Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/pucpr-br/medgemma-pt-finetuned-multiclinsum](https://huggingface.co/pucpr-br/medgemma-pt-finetuned-multiclinsum)  
25. Running MedGemma-4B on CPU or Using GGUF \+ llama-cpp | by ..., acessado em maio 21, 2026, [https://medium.com/the-owl/running-medgemma-4b-on-cpu-or-using-gguf-llama-cpp-b67e9ac4cf29](https://medium.com/the-owl/running-medgemma-4b-on-cpu-or-using-gguf-llama-cpp-b67e9ac4cf29)  
26. MedGemma-Sum-Pt: A Lightweight Model for Portuguese Clinical Summarization \- CEUR-WS.org, acessado em maio 21, 2026, [https://ceur-ws.org/Vol-4038/paper\_42.pdf](https://ceur-ws.org/Vol-4038/paper_42.pdf)  
27. pucpr-br/Clinical-BR-LlaMA-2-7B \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/pucpr-br/Clinical-BR-LlaMA-2-7B](https://huggingface.co/pucpr-br/Clinical-BR-LlaMA-2-7B)  
28. cabelo/clinical-br-llama-2-7b \- Ollama, acessado em maio 21, 2026, [https://ollama.com/cabelo/clinical-br-llama-2-7b](https://ollama.com/cabelo/clinical-br-llama-2-7b)  
29. CardioBERTpt \- Portuguese Transformer-based Models for Clinical Language Representation in Cardiology \- GitHub, acessado em maio 21, 2026, [https://github.com/HAILab-PUCPR/CardioBERTpt](https://github.com/HAILab-PUCPR/CardioBERTpt)  
30. pucpr-br/cardiobertpt \- Hugging Face, acessado em maio 21, 2026, [https://huggingface.co/pucpr-br/cardiobertpt](https://huggingface.co/pucpr-br/cardiobertpt)  
31. Class of LLMs: Benchmarking Large Language Models on the Brazilian National Medical Examination \- ACL Anthology, acessado em maio 21, 2026, [https://aclanthology.org/2026.propor-2.17.pdf](https://aclanthology.org/2026.propor-2.17.pdf)  
32. Performance of Large Language Models on the Brazilian National Medical Education Examination (ENAMED): Comparative Benchmark Study (Preprint) \- ResearchGate, acessado em maio 21, 2026, [https://www.researchgate.net/publication/404662417\_Performance\_of\_Large\_Language\_Models\_on\_the\_Brazilian\_National\_Medical\_Education\_Examination\_ENAMED\_Comparative\_Benchmark\_Study\_Preprint](https://www.researchgate.net/publication/404662417_Performance_of_Large_Language_Models_on_the_Brazilian_National_Medical_Education_Examination_ENAMED_Comparative_Benchmark_Study_Preprint)  
33. Class of LLMs: Benchmarking Large Language Models on the Brazilian National Medical Examination \- ACL Anthology, acessado em maio 21, 2026, [https://aclanthology.org/2026.propor-2.17/](https://aclanthology.org/2026.propor-2.17/)  
34. Best local MEDICAL LLM models in Jan 2026? : r/LocalLLaMA \- Reddit, acessado em maio 21, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1q5pexc/best\_local\_medical\_llm\_models\_in\_jan\_2026/](https://www.reddit.com/r/LocalLLaMA/comments/1q5pexc/best_local_medical_llm_models_in_jan_2026/)  
35. K Health enhances its AI physician by fine-tuning Gemma 3 with real-world clinical data, acessado em maio 21, 2026, [https://deepmind.google/models/gemma/gemmaverse/k-health/](https://deepmind.google/models/gemma/gemmaverse/k-health/)  
36. How to create a local offline voice assistant in Python with Faster Whisper and Ollama, acessado em maio 21, 2026, [https://astanahub.com/en/blog/kak-sozdat-lokalnogo-oflain-golosovogo-assistenta-na-python-s-faster-whisper-i-ollama](https://astanahub.com/en/blog/kak-sozdat-lokalnogo-oflain-golosovogo-assistenta-na-python-s-faster-whisper-i-ollama)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA9CAYAAAAQ2DVeAAAEJklEQVR4Xu3dS8htUxwA8CWUV3mTUJ9HyiPKq4iJSAYkj5GhgYmRkrqRb6JEGeiKdFMGSEgGRBl8ZSIGRmJAoVuiuKUo5LH+1t73rL3uPt/dbufr3M75/erfOeu/19ln+m89UwIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAdtylOTZyHJHjguEjAACW7aYcz+Z4I8dbOR4YPj4kp+X4K8c/Of7Isbf7fncqReEiPJbKOyN+znHz8DEAwGr4OMeDVfuaVIqt7VzRJubYlePvJrc7x74m15r6/vB9jq02CQCwSv7McVfVviodfAQs+kzxbioFVe32VEbEtjP1/SHe9USbBABYJf1U5Zc5LmyezTO1oIrRtbaYeigtrmA7IZWC8Pz2AQDAKro/lUIqRsVax+Q4q4pbm/bYiFz8ZqyY+jzHT00ufn9qmv/+iDF3pAMLQgCAlXJO0/4ija8HuyfHi1W83bRPmnXd7+xUir8o3Gox6ranyZ2R4+k0//0RY6JYu61NAgCskthwUPsmDdezzTNlyjI2HLS7NjdzXNvkxkx5f4gRvJgWBQBYWbHhoJ/OPDaVkayjZ4/nmlJQvZ9m06HxH5fk+Gr2eFtT3h8OthYOAFhjcexFf/5XjPJEgfFZlbuo69e33+na4cMqX8fXOc7McVyOV7pcnF/2Y44fcjwXP16wKM42ctyZyn9PNbWgOlQ7/X4AYE3clw4c4Yl25HuPVt9r0e+RJvdqGq73at8d/WOHJQAAE7XniW107boQe6r6XoupyBu677E4PzzTffaiTy+mFGPUbRE3EAAArI2Ytvu1aj+ZSsH2cte+MY0fdxGn+L+X49xUdjh+Mnz8n4tzvFm1Yxp1rB8AANvoC7bjU1kLFofORjsKtti5+Pys60BMmca9l9+lskZt7NyzOF8sFuh/lOO3VHZbHjnoMRPvmRex8B8AYG3FYa7fplKovdTl4kDYrVQ2CMw7buLTNFvgH3d4RjF2YpWL321V7fBL+n/3a07RbnpYlQAA2O/kVIqvx3Pc2+WigItRsVv6TiNiFO6oJleffRbToVH49X3if8ZuDOi1NwLUEQfSAgCsrZgKjbVlsVatt5WG69pasaat3UxweY7XqtzuNOxzfSqFYBRgL1R5AAAmaM8LuzLHKU1uEa5O5XooAABYiP5g4X05Lqvy/TqzvVUOAIAl2UylMIsL5XuxcaJdvwcAwJLsSWU9Xr1zs74FAgCAJXs4lQvl4yy467rc67PHAAAsU+yg7a/civPkYmr0vOQuVACAw0acGddeZB+HBPf3pwIAsGSbTbtfy2bDAQDAYeD0HB80uVjD9nuTAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgB32Lzm/33gwHnBGAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAaCAYAAACHD21cAAABG0lEQVR4Xu3TIUtDURjG8XeooEwUNmxahmVgUxDbkiytKDjQZhAMpo2hWEUEw5jBbLOY7DKMWzJYBEHBL7BgnPP/eO5113f3Ewwf+MF433POPfecO7OxyCxK2MR8VMtjIR7gM4VTvOIIh+jiEo8oDocOM4lr3CKbqOtJHbQt7GQkG3jHim+QE7R8Mc4ZPrDoG6SOii/GucEAx5hwvWXkXO03VQsTpY8H7GMuOSgtOtFzC5PiBeTJ0rc/Em1Tx36BnoXJB39GRNFAvUPGN0gZX2j4hlLAlYV79FnFJ3Z9Q9Ex32PaN8geXrDkG4ruT6uuu7q2r4PZcvWf6BO6Qw3P0W9dQRNv2LH0d7cZCysruo41bFv4Z6Rt/T9jlG+J/Ct4t2V5pAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA+UlEQVR4Xu3Tv0sCcRjH8UdQUFRcBIWcGgShLdSlhsA5on/C/0caW9oaWoSChgbRv0FcKwJByKYCk9L38Xi/HtPLzeE+8ILjPndf+D53X5E4u+YMYywCppisrr9xj6r7QlSuMceJuV9BFx9omG4teQwwRNF0TkoY4REZ04VSwzvukDSdmxvRZ5xnN+ZcdB5tWwTiLPSFui2C6cjf83GTxRM+cWw6Lzn0ZPN8nBzgWfTrHoYrP/+ZTwu/eEDadF6i5pPAlehCl6bzEvXZy+iLzqdgulCORH80u60ULkTncitbFjnFi/hH4gdveBU9GjPRY9EU3VqcOPufJU75NY0EHNwQAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAc0lEQVR4XmNgGPogEojfAfF/JPwFiDOQFSEDRiCeD8T/gNgFTQ4DCALxaSB+AMTSqFKYwBiIvwLxGiBmQZPDANEMELcWoUugA5h7fwOxDZocBoC59y4Qi6PJYQDauJf6QRYLxM8YUKP4FRAnIysaBYMUAAD9Px2F6V8OKAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABHElEQVR4XmNgGAWUAkcgfg3E/6F4BxBzIsnzAfEuJHkQXgfE3EhqUAAjEM8C4l9A/BOILVGlwSAIiNcwoFqEFQgC8UIgzmeA2DyFAWIBMigC4mg0MaxAH4j7gVgSiK8D8RMgVkSSZwHi2VB1BAHIxnQou4EB4rocuCwDgwgDxOUgHxAEfUBsDGXrAPF7ID4BxPxQMRsgngxl4wWw8ALZDgIgLy0H4n9A7AEVA7mapPBCDnCQISDDQIaCYo+s8IIBkPdA3gR514mByPACuQYUFqboEkAQwwCJiGtA3IkmhxWghxcyEGeAJBOQgUSFF8gLoKzBhS4BBQ1A/BaINdHEUYALEH9hQOQ1UBbyRlEBAaBkAsqrBMNrFIyCIQMA260zNBT6yKgAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAADFElEQVR4Xu2XS6hOURTH/0KRZ5G35JESmXgMkAw8ByQDKY+uFN0ykEfyTElESsojocyIMhID6RPJSIiUUld5JMlMIY//v7VXZ599v/N958otbudfv849++y7ztprr7X2+YBKlSpV+rc0m6wnPdMHXVFDySvynAxOnnVJdSOzyIT0QaWg7mQSWUj6wOpgXG5Gc/UnS8jocC87sjcNZt81HDZvEGxnJpKlsP9P1YvMI8uR2U2lcbdXSnrhQ3KYbCR3ySNyKJ7URL3JKbKbvCNHyDWyOlzPwYI4ilwkx2E1dxr2nmPkHhkAk+buII9hPsnOE7INFiRJ1y3kDNlP3pLJ4VmhxpCXZBcyQy3kF1kW7stoPtkDy5JP5DwsCNIM8jnM2UQWw3ZE79B7teNxw9FitYgXMP9cK2B2poT7BWQvzG/Z/gnLlEL1gDn2hoyNxnfCnJbzZdVCZsKC9APmgGsO+U62k32wBSoT3sPKRgtcS6aH+VqYbKwJ9y4tRkHyRSkDPCBbUcJn3w2lnBYv6ar7GukbxjoipWcbGRmNaVflqBYiqS5vBPR3LH/mwYjldtJddJ9VEv2SZzl5xGTIJUfbyMlorKzUpG6jfgDj6KuOlVXa5VTa/ddoHwy3k2ajpDJQOTT12ResenIp/b6iY/Xr0o5oZ1QSLjknJ1WTHgSl+7dwTeUfEJeScTWjj+QEsl7j8pJRY2soFb8i74tTh9QONa2FAsmOAngg3Muxg+QZ8s0nrt9U3lduIWt6ypzL5CayLh5LGVrKZzm0GdbuL8COIx0pNfx5/aqL6oi7CrOnNBwWzdF5fIVcR/v6dSnl78NqUn49hTUoD0Cs0vUbS4sbQkbA0imuhQ2wM7QRckZ1VIPVnhzTR0BR0DRetFiXNkM25Ff80ZLKS6Bp/daT14J3U2kgrJE0Qo6NR/v67SzpV5HOX6W3znT1A11LS+egInWUfCGLULw79aS5+hpS/a6DBamzpHfVyAfYj4U7yL7iSmsqOYt8ms7NzWisNO1X5R//VSnVW2F95wFZicYpX6lSpUr/p34Dl7ubYqZ383wAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAaCAYAAAAEy1RnAAADBUlEQVR4Xu2WS6hPURSHl1BEHpFHyCUpDBAG8kjyyIDERBkpr3QZEPIY3MSAUB4jkUehUEhiYHAHBopS8hhIHokMpBSFxO9r7X2ds/+Pe/6De9PtfPV17zl7/8/Ze6219z5mJSUlJf83PeUaOStt6MoskX/k8bShK9NHLpT90oYS0V1Ok3PN1wR0k5Pl6NipAYabl9mgcM3fpXK8+XOBd842fy//k6FF5u+MfbIMNX8GfeibUm0ONaHDEbldPgj/Ay//Ku9Z9ZfUYog8K/fJN/KoPC1Xy/tyh/mkNoc+z+UZeU2uk0/kKvvHsNB2PdxnnK/l9Eyf/vKS3Cpvyluyd6a9Aup/rxxoPunz4T4/OidbZd9wrwgMnMEtkL/lLvMswAr52TzDx+Rg86C+kk3mg2YT2hD6U2VP5SnLV+BFedd8jFzvl3NCOxvYW/NqqwmDnCBnyu+WjzKDO5G5LsJu88HuNJ8gz46QbSa1Ra6VI+V7eSC0U/7rzSurh3mFfJDjQnuExMSJEbg95v35HUG8LXu19a5Di/kAxmTukZkY9UZgAJRkq+WrhCwQ2Bnhmmr4Gf6mECyCxnN4XoTJMKlq2RwrP5oHvl0o7YeWfwFl02L5TBWFTYeSjRmE+I5n5tkBBkegyXgKm1a21CNxYpctHwyoF8QKiBiRoyQjZPywVT64CCyLH3JZ5l5cPtvCdcxYrVKMHxlMPstG+ct8L0ohyASEwLRLzEwsCzaNg3JqW4/GIHgMOO4PPO+KvGO+00K6nlNGyZeyOXNvkvk4CVx6rLUXxAp4wCb5SV4w3xmX53oUJ67nd/KR+abz2Hw9Z48+Mv9FzsvcSyHbBOZqkOUx3yonDA2t5yxEiFJPI9VkfsRwdNRzihxhfj4zSY4qzu30eUAb67zaBLJQJVTigLQhgQBxRBZaz0WILyYg9WRy1dZzR8CYOG45BgngSfnCfJydCtk4ZL5hLbbGPmoahU/Ob+ZfapQ8pb0y16MTIMt8ImbLne/gjoINMe4XN+TEfHNJSUlJF+YvEciO5f1uW0kAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAABm0lEQVR4Xu2VPyhFURzHv5Ii//IvKRJZSBlMyixKKSmhrFI2g5gYDIwWschgVZKSDKJMSgYGmWS1KJOB7/f+3uvde9/x5y73PXU+9en1zj2ne3/nfM85gMfjKUYq6DTdpZu0h5ZEehQxtfSQztIW2kcv6SL+SRELdCXW1ktvaUesvSjZpxuIzrZWQgVoNZzU0BHalvmv31HYwLRZp590i1Zm2ibpCa3Kdgqjxh26Rl9gA7fpEn1A+svWTh9hRTzBCrrKtDsZpnOw5Xmje7SOntNX2p3rmscUfU7gDe0KRv6M3qn+KkIqVkqJk3nYx4/TDzoIy58ipOLS3vmtsBnXpOokeocVcU0bQv3yUHTuaWP8QYro/D+GxTeLCjqCFaETykk1rOoDJJvxcthG/6vNtCwY6UZJUP7jsVVhp7AoOdEA5V3LlgRtrIkEjtH6YKSbftjB0Rl/ALsblBIn4fwXEp2IZ3QZ0SQ00Qs6EGqLsErv8MsmSQmdUvoWFTIDm3ntTW3ob+OtLDsviQJRCouTYjeE3IXm8Xg8Hk+EL8v5S6UIIlg6AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAaCAYAAAAEy1RnAAACGUlEQVR4Xu2WzUsVURjGn9BQUSpMszCRoARp0UKXoSEtdBFtBIsgo00uhCARofxHikRbFSi18INwm4sICYKCMEQKW4oQtevD5+k9gzNzZ5qZjdOF84Mf13vOjMzzzjnvuYDH4/GUQwO9RY/FxsMcoXfpI3rffa86mukInaO79DM9Fb4gxAD9RG/Crpmkz2HFqioU+irtpc+QHvo8/QorkGijm0i/vmp4guQQtfQx/UBb3NgheoPedvOZqEpX6CVaH50qlbTQZ+g2XaB19AQ9DgueyUnYjS/oNfoA9s+0tEQ7Pef+LoO00Jfpb7pEH9J79Cldo6dD11XQSd/Dut5hN6ZloZtfwprBNL3o5pK4Tr8UcJ2e/XtnPtJCa1X+oT9ojxtTBr28VdrkxiIEe0JvVUslzBT9RgfpLD0anT5QskIvI7od9exaAVoJFXTTHdjSjm96NQNVUIGDzlgWaaGHYKE1H0ahNa7PCoJK3YlPYH9uEdnnnaqsB8qrGmawlfKQFvoCbDUWCh1USgHjaOwn7Y9PJKC+MFxAnb86h/OSFlpb7jUqV+o/l3cH7JfMeGhM7V6/cD5ivyDqhME5WAYKrb6T1JHH6AYsi8hsZEIBt+g8nYFVboK20hX6BtbF9TYPEp25r+h3WPHlL1h4BQ1ohPWdt3QUduq8Q47nrUHywa5xveEi+68M9MxdsK3Th///eT0ej8fj8SSwB9MUdaFapH6AAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAZCAYAAAAmNZ4aAAABVklEQVR4Xu2VPytGYRiHb4OilGSQMmAggygZGEwWA6nXIHwCWVFWWXwBJYvBR2BRGJWJwkzKZGOg/Ll+nXPqOc85p/MoXoNz1dXb+Z2783ve09v9mlX8J5pxAXdxG/vTt79NN877oU8rHuMmtuAw3mDNHQpgAJfxBN9xP307yzpeYJuTLeItdjhZGSqexXF8sJJilanUHxrFZ5zx8hA68c6yz0yhUz5ZdmgEX3DLy0MIKk4K/KGiPISg4mn8tOzQrxdP2R8VFxUU5SEEFffio2WHkuINLw8hqFgL4wwPscnJJ/Et/kzQbDs2OFkeQcViCe+xJ77Wg7XFzi3aakKFl/iKY3FWRFJ8YCWHbMQdPLVo86j02qLVmaBve4QfuOrkLno72lhal/rBSi2hKxx05lLoZH04hxMWHSYPPXzFD+vBmpW/6h9nCPcs+gutK9pyXX5YUZHHF2JvShfKJ26dAAAAAElFTkSuQmCC>