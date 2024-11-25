from flask import Flask, request, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document  # Import necessário para criar os objetos Document
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, render_template, redirect


# Configuração do modelo LLM
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

template = """
Compartilharei uma pergunta com você, e você me dará a melhor resposta que devo enviar para o paciente, com base nas melhores respostas anteriores, seguindo TODAS as regras abaixo:

1/ A resposta deve ser muito semelhante ou até mesmo idêntica às melhores respostas anteriores, em termos de comprimento, tom de voz, argumentos lógicos e outros detalhes.

2/ Se as melhores respostas forem irrelevantes, tente imitar o estilo das melhores respostas para a mensagem do paciente.

3/ Sua resposta deve ser sempre em portugues do Brasil.

Abaixo está a pergunta que recebi do paciente:
{message}

Aqui está uma lista de melhores respostas de como normalmente respondemos a pacientes em cenários semelhantes:
{best_practice}

Por favor, escreva a melhor resposta que devo enviar para este paciente:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Carrega as variáveis de ambiente
load_dotenv()

# Caminhos
faiss_index_path = "faiss_index"
csv_path = "train.csv"

app = Flask(__name__)

# Inicializa o índice FAISS
if not os.path.exists(faiss_index_path):
    # Carregue o CSV
    df = pd.read_csv(csv_path)

    # Extraia a coluna desejada
    problemas = df["Question"].tolist()

    # Converta cada string em um objeto Document
    documents = [Document(page_content=issue) for issue in problemas]

    # Crie embeddings e base FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

    # Salve o índice FAISS no disco
    db.save_local(faiss_index_path)
else:
    # Carregue o índice FAISS salvo
    db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        # Faça a pesquisa de similaridade
        results_with_scores = db.similarity_search_with_score(query, k=3)
        
        # Defina o limite de similaridade
        similarity_threshold = 0.3

        # Filtre os resultados baseados na pontuação
        results = [
            {"content": doc.page_content, "score": score}
            for doc, score in results_with_scores
        ]

    return render_template("index.html", results=results, query=query)

@app.route("/add", methods=["GET", "POST"])
def add_question():
    message = ""

    if request.method == "POST":
        question = request.form.get("question")
        answer = request.form.get("answer")

        if question and answer:
            # Carregue o arquivo CSV
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                # Caso o arquivo não exista, inicialize um DataFrame vazio
                df = pd.DataFrame(columns=["Question", "Answer"])

            # Adicione a nova pergunta e resposta ao DataFrame
            new_row = {"Question": question, "Answer": answer}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Salve de volta no CSV
            df.to_csv(csv_path, index=False)

            # Atualize o índice FAISS com a nova pergunta
            document = Document(page_content=question)
            db.add_documents([document])
            # Salve o índice FAISS no disco
            db.save_local(faiss_index_path)

            message = "Pergunta e resposta adicionadas com sucesso!"
        else:
            message = "Por favor, preencha todos os campos."

    return render_template("add.html", message=message)

@app.route("/ask", methods=["GET", "POST"])
def ask_question():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        # Realize a pesquisa de similaridade
        results_with_scores = db.similarity_search_with_score(query, k=3)  # Apenas os 10 primeiros resultados

        # Prepare os resultados para exibição
        for doc, score in results_with_scores:
            # Encontre a resposta correspondente no CSV
            df = pd.read_csv(csv_path)
            answer = df.loc[df["Question"] == doc.page_content, "Answer"].values[0]
            results.append({"question": doc.page_content, "answer": answer, "score": score})

    return render_template("ask.html", results=results, query=query)

@app.route("/ask_chatgpt", methods=["GET", "POST"])
def ask_chatgpt():
    gpt_response = ""
    query = ""

    if request.method == "POST":
        query = request.form.get("query")

        # Realize a pesquisa de similaridade
        results_with_scores = db.similarity_search_with_score(query, k=3)

        # Extraia as três perguntas mais relevantes
        best_practices = [
            doc.page_content for doc, score in results_with_scores
        ]

        # Combine as melhores práticas em uma string
        best_practice_text = "\n".join(best_practices)

        # Envie as práticas e a mensagem para o modelo
        gpt_response = chain.run({
            "message": query,
            "best_practice": best_practice_text
        })

    return render_template("ask_chatgpt.html", response=gpt_response, query=query)

@app.route("/table", methods=["GET"])
def show_table():
    # Carregue o CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Question", "Answer"])  # Caso o CSV não exista ainda

    # Converta o DataFrame para uma lista de dicionários para renderizar no HTML
    table_data = df.to_dict(orient="records")

    return render_template("table.html", table_data=table_data)

@app.route("/delete/<int:index>", methods=["POST"])
def delete_entry(index):
    # Verifique se o CSV existe
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Verifique se o índice é válido
        if 0 <= index < len(df):
            # Remova a entrada do DataFrame
            df.drop(index, inplace=True)
            df.reset_index(drop=True, inplace=True)  # Reindexa após a exclusão

            # Salve o CSV atualizado
            df.to_csv(csv_path, index=False)

            # Recrie o índice FAISS com o novo conjunto de perguntas
            questions = df["Question"].tolist()
            documents = [Document(page_content=question) for question in questions]
            global db  # Atualize a variável global db
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(documents, embeddings)
            # Salve o índice FAISS no disco
            db.save_local(faiss_index_path)

    return redirect("/table")




if __name__ == "__main__":
    app.run(debug=True)
