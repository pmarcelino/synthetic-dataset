import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

num_threads = 25


def extract_text_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def process_text(text, min_length=250):
    lines = text.split("\n")
    filtered_lines = [
        line
        for line in lines
        if len(line.strip()) > min_length and line.strip().endswith(".")
    ]
    return "\n".join(filtered_lines)


def read_text_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def write_to_file(content, file_path):
    with open(file_path, "w") as f:
        f.write(content)


def get_llm_response(user_prompt, system_prompt, model="gpt-3.5-turbo", temperature=0):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def generate_questions_and_answers(lines, question_system_prompt, answer_system_prompt):
    df = pd.DataFrame(columns=["text", "question", "answer"])

    for line in lines:
        # Generate question
        questions = get_llm_response(
            user_prompt=line, system_prompt=question_system_prompt
        ).split("\n")

        questions = [
            question.split(". ", 1)[-1] for question in questions if question.strip()
        ]

        # Generate answer
        answers = []
        for question in questions:
            user_prompt = (
                "[DOCUMENTATION]\n" + line + "\n\n---\n\n" + "[QUESTION]\n" + question
            )
            answer = get_llm_response(
                user_prompt=user_prompt, system_prompt=answer_system_prompt
            )
            answers.append(answer)

        # Save data
        new_data = {
            "text": [line] * len(questions),
            "question": questions,
            "answer": answers,
        }

        df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

    return df


def evaluate_questions_gpt4(row) -> pd.DataFrame:
    text = row["text"]
    question = row["question"]

    # Get additional questions
    additional_questions_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A questions for a given documentation.
    """

    additional_questions_user_prompt = f"""
    Please propose at most three concise questions about whether a potential Question is a good Q&A question for a given documentation. Another assistant will evaluate different aspects of the Question by answering all the questions.

    Here are some rules of the evaluation:
    (1) You should prioritize evaluating whether the Question honestly/precisely/closely relates to the documentation.
    (2) Questions should NOT contain more/less than what the documentation provides.

    # Documentation:
    {text}
    
    # Question:
    {question}

    # Requirements for Your Output:
    (1) The questions should **specifically** target the given Question instead of some general standards, so the questions may revolve around key points of the Question.
    (2) You should directly give the questions without any other words.
    (3) Questions are presented from most important to least important.
    """

    additional_questions = get_llm_response(
        additional_questions_user_prompt,
        additional_questions_system_prompt,
        model="gpt-4",
    )

    print(additional_questions)

    # Get question from a stronger AI assistant
    question_from_stronger_ai_system_prompt = """
    You are a strong AI assistant generating *one and only one* question for a given documentation.
    """

    question_from_stronger_ai_user_prompt = f"""
    {text}
    """

    question_from_stronger_ai = get_llm_response(
        question_from_stronger_ai_user_prompt,
        question_from_stronger_ai_system_prompt,
        model="gpt-4",
    )

    print(question_from_stronger_ai)

    # Evaluate the question
    llm_evaluator_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A questions for a given documentation. Your goal is to evaluate the questions for the given documentation.
    """

    llm_evaluator_user_prompt = f"""
    Evaluate if the Question is useful for the given documentation. The Question was produced by an AI chatbot.

    Here are some rules of the evaluation:
    (1) The Question should be directly related to the content, purpose, or scope of the documentation. It should address a genuine need for clarification, additional information, or deeper understanding of the material presented. Questions that are off-topic or unrelated to the documentation's main objectives are less likely to be useful to the intended audience.
    (2) The Question should NOT contain more/less than what the documentation provide.
    (3) A useful question is one that is clearly and concisely formulated. It should be specific enough to elicit a targeted response, yet broad enough to be of interest to other readers or users of the documentation. Ambiguous, vague, or overly complex questions can lead to confusion and may not result in helpful answers.

    Do NOT provide any explanation for your choice.
    You should answer using ONLY "Useful" or "Not useful". Do NOT output any other words.

    # Documentation:
    {text}

    # Question:
    {question}

    # Questions about Question:
    Here are at most three questions about the Question, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
    {additional_questions}

    # A reference Question generated by a strong AI assistant:
    {question_from_stronger_ai}

    # Is the Question useful? Your response should be either "Useful" or "Not useful":
    """

    llm_evaluation = get_llm_response(
        llm_evaluator_user_prompt,
        llm_evaluator_system_prompt,
        model="gpt-4",
    )

    print(llm_evaluation)

    return llm_evaluation


def evaluate_questions_gpt4_turbo(row) -> pd.DataFrame:
    text = row["text"]
    question = row["question"]

    # Get additional questions
    additional_questions_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A questions for a given documentation.
    """

    additional_questions_user_prompt = f"""
    Please propose at most three concise questions about whether a potential Question is a good Q&A question for a given documentation. Another assistant will evaluate different aspects of the Question by answering all the questions.

    Here are some rules of the evaluation:
    (1) You should prioritize evaluating whether the Question honestly/precisely/closely relates to the documentation.
    (2) Questions should NOT contain more/less than what the documentation provides.

    # Documentation:
    {text}
    
    # Question:
    {question}

    # Requirements for Your Output:
    (1) The questions should **specifically** target the given Question instead of some general standards, so the questions may revolve around key points of the Question.
    (2) You should directly give the questions without any other words.
    (3) Questions are presented from most important to least important.
    """

    additional_questions = get_llm_response(
        additional_questions_user_prompt,
        additional_questions_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(additional_questions)

    # Get question from a stronger AI assistant
    question_from_stronger_ai_system_prompt = """
    You are a strong AI assistant generating *one and only one* question for a given documentation.
    """

    question_from_stronger_ai_user_prompt = f"""
    {text}
    """

    question_from_stronger_ai = get_llm_response(
        question_from_stronger_ai_user_prompt,
        question_from_stronger_ai_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(question_from_stronger_ai)

    # Evaluate the question
    llm_evaluator_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A questions for a given documentation. Your goal is to evaluate the questions for the given documentation.
    """

    llm_evaluator_user_prompt = f"""
    Evaluate if the Question is useful for the given documentation. The Question was produced by an AI chatbot.

    Here are some rules of the evaluation:
    (1) The Question should be directly related to the content, purpose, or scope of the documentation. It should address a genuine need for clarification, additional information, or deeper understanding of the material presented. Questions that are off-topic or unrelated to the documentation's main objectives are less likely to be useful to the intended audience.
    (2) The Question should NOT contain more/less than what the documentation provide.
    (3) A useful question is one that is clearly and concisely formulated. It should be specific enough to elicit a targeted response, yet broad enough to be of interest to other readers or users of the documentation. Ambiguous, vague, or overly complex questions can lead to confusion and may not result in helpful answers.

    Do NOT provide any explanation for your choice.
    You should answer using ONLY "Useful" or "Not useful". Do NOT output any other words.

    # Documentation:
    {text}

    # Question:
    {question}

    # Questions about Question:
    Here are at most three questions about the Question, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
    {additional_questions}

    # A reference Question generated by a strong AI assistant:
    {question_from_stronger_ai}

    # Is the Question useful? Your response should be either "Useful" or "Not useful":
    """

    llm_evaluation = get_llm_response(
        llm_evaluator_user_prompt,
        llm_evaluator_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(llm_evaluation)

    return llm_evaluation


def evaluate_answers_gpt4(row) -> pd.DataFrame:
    text = row["text"]
    question = row["question"]
    answer = row["answer"]
    llm_evaluation_question_gpt4 = row["llm_evaluation_question_gpt4"]
    llm_evaluation_question_gpt4_turbo = row["llm_evaluation_question_gpt4_turbo"]

    # Check if the question is useful
    if (
        llm_evaluation_question_gpt4 == "Not useful"
        or llm_evaluation_question_gpt4_turbo == "Not useful"
    ):
        return "Not useful"

    # Get additional questions
    additional_questions_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A answers for a given documentation and question.
    """

    additional_questions_user_prompt = f"""
    Please propose at most three concise questions about whether a potential Answer is a good Q&A answer for a given question and documentation. Another assistant will evaluate different aspects of the Answer by answering all the questions.

    Here are some rules of the evaluation:
    (1) You should prioritize evaluating whether the Answer honestly/precisely/correctly answers to the question.
    (2) Answers should NOT contain more/less than what the documentation provides.

    # Documentation:
    {text}
    
    # Question:
    {question}
    
    # Answer:
    {answer}

    # Requirements for Your Output:
    (1) The questions should **specifically** target the given Answer instead of some general standards, so the questions may revolve around key points of the Answer.
    (2) You should directly give the questions without any other words.
    (3) Questions are presented from most important to least important.
    """

    additional_questions = get_llm_response(
        additional_questions_user_prompt,
        additional_questions_system_prompt,
        model="gpt-4",
    )

    print(additional_questions)

    # Get question from a stronger AI assistant
    answer_from_stronger_ai_system_prompt = """
    You are a strong AI assistant generating a correct and detailed answer for a given question, based on a specific documentation.
    """

    answer_from_stronger_ai_user_prompt = f"""
    [DOCUMENTATION]
    {text}
    [QUESTION]
    {question}
    [ANSWER]
    """

    question_from_stronger_ai = get_llm_response(
        answer_from_stronger_ai_user_prompt,
        answer_from_stronger_ai_system_prompt,
        model="gpt-4",
    )

    print(question_from_stronger_ai)

    # Evaluate the question
    llm_evaluator_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A answers for a given question and documentation. Your goal is to evaluate the answer for the given question, based on the provided documentation.
    """

    llm_evaluator_user_prompt = f"""
    Evaluate if the Answer is correct for the given question and considering the documentation provided. The Answer was produced by an AI chatbot.

    Here are some rules of the evaluation:
    (1) The answer should directly address the question posed, using information from the provided documentation. It should not include irrelevant details or stray from the topic. The content of the answer must be factually accurate and in alignment with the information in the documentation.
    (2) The Answer should NOT contain more/less than what the documentation provide.
    (3) The answer should cover all aspects of the question, providing a comprehensive response. If the question has multiple parts or requires explanation of various concepts, the answer should address each part adequately. It should also include necessary details from the documentation to support the response, ensuring the answer is thorough and informative.

    Do NOT provide any explanation for your choice.
    You should answer using ONLY "Correct" or "Not correct". Do NOT output any other words.

    # Documentation:
    {text}

    # Question:
    {question}
    
    # Answer:
    {answer}

    # Questions about Answer:
    Here are at most three questions about the Answer, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
    {additional_questions}

    # A reference Answer generated by a strong AI assistant:
    {question_from_stronger_ai}

    # Is the Answer correct? Your response should be either "Correct" or "Not correct":
    """

    llm_evaluation = get_llm_response(
        llm_evaluator_user_prompt,
        llm_evaluator_system_prompt,
        model="gpt-4",
    )

    print(llm_evaluation)

    return llm_evaluation


def evaluate_answers_gpt4_turbo(row) -> pd.DataFrame:
    text = row["text"]
    question = row["question"]
    answer = row["answer"]
    llm_evaluation_question_gpt4 = row["llm_evaluation_question_gpt4"]
    llm_evaluation_question_gpt4_turbo = row["llm_evaluation_question_gpt4_turbo"]

    # Check if the question is useful
    if (
        llm_evaluation_question_gpt4 == "Not useful"
        or llm_evaluation_question_gpt4_turbo == "Not useful"
    ):
        return "Not useful"

    # Get additional questions
    additional_questions_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A answers for a given documentation and question.
    """

    additional_questions_user_prompt = f"""
    Please propose at most three concise questions about whether a potential Answer is a good Q&A answer for a given question and documentation. Another assistant will evaluate different aspects of the Answer by answering all the questions.

    Here are some rules of the evaluation:
    (1) You should prioritize evaluating whether the Answer honestly/precisely/correctly answers to the question.
    (2) Answers should NOT contain more/less than what the documentation provides.

    # Documentation:
    {text}
    
    # Question:
    {question}
    
    # Answer:
    {answer}

    # Requirements for Your Output:
    (1) The questions should **specifically** target the given Answer instead of some general standards, so the questions may revolve around key points of the Answer.
    (2) You should directly give the questions without any other words.
    (3) Questions are presented from most important to least important.
    """

    additional_questions = get_llm_response(
        additional_questions_user_prompt,
        additional_questions_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(additional_questions)

    # Get question from a stronger AI assistant
    answer_from_stronger_ai_system_prompt = """
    You are a strong AI assistant generating a correct and detailed answer for a given question, based on a specific documentation.
    """

    answer_from_stronger_ai_user_prompt = f"""
    [DOCUMENTATION]
    {text}
    [QUESTION]
    {question}
    [ANSWER]
    """

    question_from_stronger_ai = get_llm_response(
        answer_from_stronger_ai_user_prompt,
        answer_from_stronger_ai_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(question_from_stronger_ai)

    # Evaluate the question
    llm_evaluator_system_prompt = """
    You are a helpful assistant in evaluating the quality of Q&A answers for a given question and documentation. Your goal is to evaluate the answer for the given question, based on the provided documentation.
    """

    llm_evaluator_user_prompt = f"""
    Evaluate if the Answer is correct for the given question and considering the documentation provided. The Answer was produced by an AI chatbot.

    Here are some rules of the evaluation:
    (1) The answer should directly address the question posed, using information from the provided documentation. It should not include irrelevant details or stray from the topic. The content of the answer must be factually accurate and in alignment with the information in the documentation.
    (2) The Answer should NOT contain more/less than what the documentation provide.
    (3) The answer should cover all aspects of the question, providing a comprehensive response. If the question has multiple parts or requires explanation of various concepts, the answer should address each part adequately. It should also include necessary details from the documentation to support the response, ensuring the answer is thorough and informative.

    Do NOT provide any explanation for your choice.
    You should answer using ONLY "Correct" or "Not correct". Do NOT output any other words.

    # Documentation:
    {text}

    # Question:
    {question}
    
    # Answer:
    {answer}

    # Questions about Answer:
    Here are at most three questions about the Answer, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
    {additional_questions}

    # A reference Answer generated by a strong AI assistant:
    {question_from_stronger_ai}

    # Is the Answer correct? Your response should be either "Correct" or "Not correct":
    """

    llm_evaluation = get_llm_response(
        llm_evaluator_user_prompt,
        llm_evaluator_system_prompt,
        model="gpt-4-1106-preview",
    )

    print(llm_evaluation)

    return llm_evaluation


# Extract data from the web
run_extract_text_from_html = False
if run_extract_text_from_html:
    url = "https://docs.llamaindex.ai/en/stable/getting_started/concepts.html"
    text = extract_text_from_html(url)
    filtered_text = process_text(text)
    write_to_file(filtered_text, "docs.txt")

# Generate questions and answers
run_questions_and_answers_generation = True
if run_questions_and_answers_generation:
    question_system_prompt = """
    You are an expert user extracting information to quiz people on documentation. You will be passed a text extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text. Present each question in a new line.
    """
    answer_system_prompt = """
    You are an expert user answering questions. You will be passed a text extracted from a documentation and a question. Generate a comprehensive and informative answer to the question based *solely* on the given text.
    """

    lines = read_text_file("docs.txt")
    df = generate_questions_and_answers(
        lines, question_system_prompt, answer_system_prompt
    )

    # Write questions and answers to a CSV file
    df.to_csv("dataset.csv", index=False)

# Evaluate questions
run_questions_evaluation = True
if run_questions_evaluation:
    df = pd.read_csv("dataset.csv")

    # Use GPT4 to evaluate questions
    with ThreadPoolExecutor(num_threads) as executor:
        results = list(
            executor.map(evaluate_questions_gpt4, df.to_dict(orient="records"))
        )

    for index, result in enumerate(results):
        df.at[index, "llm_evaluation_question_gpt4"] = result

    # Use GPT4 Turbo to evaluate questions
    with ThreadPoolExecutor(num_threads) as executor:
        results = list(
            executor.map(evaluate_questions_gpt4_turbo, df.to_dict(orient="records"))
        )

    for index, result in enumerate(results):
        df.at[index, "llm_evaluation_question_gpt4_turbo"] = result

    df.to_csv("dataset_with_q_evaluation.csv", index=False)

# Evaluate answers
run_evaluate_answers = True
if run_evaluate_answers:
    df = pd.read_csv("dataset_with_q_evaluation.csv")

    # Use GPT4 to evaluate answers
    with ThreadPoolExecutor(num_threads) as executor:
        results = list(
            executor.map(evaluate_answers_gpt4, df.to_dict(orient="records"))
        )

    for index, result in enumerate(results):
        df.at[index, "llm_evaluation_answer_gpt4"] = result

    # Use GPT4 Turbo to evaluate answers
    with ThreadPoolExecutor(num_threads) as executor:
        results = list(
            executor.map(evaluate_answers_gpt4_turbo, df.to_dict(orient="records"))
        )

    for index, result in enumerate(results):
        df.at[index, "llm_evaluation_answer_gpt4_turbo"] = result

    df.to_csv("dataset_with_qa_evaluation.csv", index=False)
