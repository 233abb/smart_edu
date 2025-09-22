import json
from json import JSONDecodeError
import time
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph
SearchType = "hybrid"  # Placeholder for the actual import if needed

from configuration import config
import os
from langchain.chat_models import init_chat_model

load_dotenv()
proxy_url = "http://127.0.0.1:10808"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")


class ChatService:
    def __init__(self):
        # --- FIX: __init__ is now restored to its original logic ---
        self.llm = init_chat_model(
            model='gemini-2.5-flash',
            model_provider="google_genai",
            temperature=0.2
        )
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
        )
        self.neo4j_vectors = self._initialize_neo4j_vectors()
        self.graph = Neo4jGraph(url=config.NEO4J_CONFIG["uri"],
                                username=config.NEO4J_CONFIG['user'],
                                password=config.NEO4J_CONFIG['password'])
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    # Unchanged methods _create_vector_store and _initialize_neo4j_vectors
    def _create_vector_store(self, entity_name: str) -> Neo4jVector:
        print(f'Initializing vector store for: {entity_name}')
        return Neo4jVector.from_existing_index(
            self.embedding_model,
            url=config.NEO4J_CONFIG["uri"],
            username=config.NEO4J_CONFIG['user'],
            password=config.NEO4J_CONFIG['password'],
            index_name=f'{entity_name.lower()}_vector_index',
            keyword_index_name=f'{entity_name.lower()}_full_text_index',
            search_type=SearchType,
        )

    def _initialize_neo4j_vectors(self) -> dict:
        entity_map = {
            'Category': 'category', 'Subject': 'subject', 'Course': 'course',
            'Teacher': 'teacher', 'Chapter': 'chapter', 'Video': 'video',
            'Question': 'question', 'KnowledgePoint': 'knowledge_point',
            'User': 'user'
        }
        return {
            entity_label: self._create_vector_store(index_prefix)
            for entity_label, index_prefix in entity_map.items()
        }

    def _generate_cypher_enhanced(self, question: str):
        # 少样本示例
        examples = [
            # --- A. 结构性/从属关系示例 ---
            {
                "question": "查找“后端开发”分类下的所有学科。",
                "output": """{"cypher_query": "MATCH (s:Subject)-[:BELONGS_TO]->(c:Category {name: $param_0}) RETURN s.name", "entities_to_align": [{"param_name": "param_0", "entity": "后端开发", "label": "Category"}]}"""
            },
            {
                "question": "“Java”学科下有哪些课程？",
                "output": """{"cypher_query": "MATCH (c:Course)-[:BELONGS_TO]->(s:Subject {name: $param_0}) RETURN c.name", "entities_to_align": [{"param_name": "param_0", "entity": "Java", "label": "Subject"}]}"""
            },
            {
                "question": "“Java从入门到精通”这门课有哪些章节？",
                "output": """{"cypher_query": "MATCH (chap:Chapter)-[:PART_OF]->(c:Course {name: $param_0}) RETURN chap.name", "entities_to_align": [{"param_name": "param_0", "entity": "Java从入门到精通", "label": "Course"}]}"""
            },
            {
                "question": "属于“高并发编程”课程的试卷有哪些？",
                "output": """{"cypher_query": "MATCH (p:Paper)-[:BELONGS_TO]->(c:Course {name: $param_0}) RETURN p.title", "entities_to_align": [{"param_name": "param_0", "entity": "高并发编程", "label": "Course"}]}"""
            },
            {
                "question": "查找“第一章：环境搭建”这个章节的视频。",
                "output": """{"cypher_query": "MATCH (:Chapter {name: $param_0})-[:HAS_VIDEO]->(v:Video) RETURN v.name", "entities_to_align": [{"param_name": "param_0", "entity": "第一章：环境搭建", "label": "Chapter"}]}"""
            },

            # --- B. 实体关联关系示例 ---
            {
                "question": "张老师教了哪些课？",
                "output": """{"cypher_query": "MATCH (t:Teacher {name: $param_0})<-[:TAUGHT_BY]-(c:Course) RETURN c.name", "entities_to_align": [{"param_name": "param_0", "entity": "张老师", "label": "Teacher"}]}"""
            },
            {
                "question": "“MySQL性能优化”课程覆盖了哪些知识点？",
                "output": """{"cypher_query": "MATCH (c:Course {name: $param_0})-[:COVERS_KNOWLEDGE]->(kp:KnowledgePoint) RETURN kp.name", "entities_to_align": [{"param_name": "param_0", "entity": "MySQL性能优化", "label": "Course"}]}"""
            },
            {
                "question": "“期中考试”试卷里有哪些题目？",
                "output": """{"cypher_query": "MATCH (p:Paper {title: $param_0})-[:HAS_QUESTION]->(q:Question) RETURN q.text", "entities_to_align": [{"param_name": "param_0", "entity": "期中考试", "label": "Paper"}]}"""
            },

            # --- C. 用户行为关系示例 ---
            {
                "question": "用户“小明”收藏了哪些课程？",
                "output": """{"cypher_query": "MATCH (u:User {name: $param_0})-[:FAVORS]->(c:Course) RETURN c.name", "entities_to_align": [{"param_name": "param_0", "entity": "小明", "label": "User"}]}"""
            },
            {
                "question": "用户“小红”看过的视频有哪些？",
                "output": """{"cypher_query": "MATCH (u:User {name: $param_0})-[:WATCHED]->(v:Video) RETURN v.name", "entities_to_align": [{"param_name": "param_0", "entity": "小红", "label": "User"}]}"""
            },
            {
                "question": "查找“小明”参加过的所有考试。",
                "output": """{"cypher_query": "MATCH (u:User {name: $param_0})-[:TOOK_EXAM]->(p:Paper) RETURN p.title", "entities_to_align": [{"param_name": "param_0", "entity": "小明", "label": "User"}]}"""
            },
            {
                "question": "“小红”回答过的问题是什么？",
                "output": """{"cypher_query": "MATCH (u:User {name: $param_0})-[:ANSWERED]->(q:Question) RETURN q.text", "entities_to_align": [{"param_name": "param_0", "entity": "小红", "label": "User"}]}"""
            },

        ]
        prompt_template = """
        你是一个专业的 Neo4j Cypher 查询生成器。你的任务是：根据用户问题和知识图谱的结构，生成一条参数化的 Cypher 查询。

        **知识图谱结构信息:**
        {schema_info}
        
        少样本示例:
        {examples}
        **任务:**
        现在，请根据下面的用户问题生成Cypher查询和需要对齐的实体。
        用户问题："{question}"

        **要求:**
        1.  仔细分析用户问题，特别是实体类型（例如，用户是想问关于一个“课程”还是一个“知识点”）。
        2.  生成参数化的Cypher查询语句，参数必须以美元符号 `$` 开头，例如 `$param_0`, `$param_1`。
        3.  识别问题中需要与图谱中的节点进行对齐的实体。
        4.  必须严格使用以下JSON格式输出结果，不要包含任何额外的解释或注释。
        ```json
        {{
          "cypher_query": "生成的Cypher语句",
          "entities_to_align": [
            {{
              "param_name": "param_name",
              "entity": "原始实体名称",
              "label": "节点类型"
            }}
          ]
        }}
        ```
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.json_parser

        try:
            # --- FIX: Directly pass the self.graph.schema string to the prompt ---
            return chain.invoke({
                "question": question,
                "schema_info": self.graph.schema,  # This is the correct way
                "examples": examples
            })
        except (JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM output for Cypher generation: {e}")
            return None

    # The rest of the enhanced methods remain unchanged as they were correct.
    def _entity_align_enhanced(self, entities_to_align: list, question: str):
        """
        Performs entity alignment using a recall-and-rank approach with LLM for disambiguation.
        """
        aligned_entities = []
        for entity_to_align in entities_to_align:
            label = entity_to_align['label']
            entity_name = entity_to_align['entity']

            try:
                candidates = self.neo4j_vectors[label].similarity_search(entity_name, k=3)
            except KeyError:
                print(f"Warning: No vector store found for label '{label}'. Skipping alignment.")
                continue

            if not candidates:
                print(f"Warning: No candidates found for entity '{entity_name}' with label '{label}'.")
                return None

            candidate_names = [c.page_content for c in candidates]

            prompt_template = """
            根据用户原始问题，从下面的候选列表中选择最相关的实体。

            用户原始问题: "{question}"
            需要匹配的实体: "{entity_name}"
            候选列表: {candidate_list}

            请直接返回最匹配的候选实体的名字。如果都不匹配，请返回 "None"。
            """
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            best_match = chain.invoke({
                "question": question,
                "entity_name": entity_name,
                "candidate_list": candidate_names
            })
            if candidate_names:
                print(candidate_names)
            if best_match != "None" and best_match in candidate_names:
                print(f"Aligned '{entity_name}' to '{best_match}'")
                aligned_entities.append({
                    "param_name": entity_to_align['param_name'],
                    "entity": best_match
                })
            else:
                print(best_match)
                print(f"Could not reliably align entity '{entity_name}'. Aborting.")
                return None

        return aligned_entities

    def _generate_answer_enhanced(self, question: str, cypher_query: str, query_result: list):
        """
        Generates a natural language answer using the question, the query, and preprocessed results.
        """
        processed_result_str = json.dumps(query_result, ensure_ascii=False, indent=2,default=str)

        prompt_template = """
        你是一个智能教学助手。请根据用户问题、用于查询知识图谱的Cypher语句以及查询结果，生成一个友好、自然的中文回答。

        ### 用户问题:
        {question}

        ### 用于检索信息的Cypher查询:
        ```cypher
        {cypher_query}
        ```

        ### 检索到的信息 (JSON格式):
        {query_result}

        ### 回答要求:
        1.  用流畅、对话式的中文进行回答，直接回答用户的问题。
        2.  如果检索到的信息为空 (`[]`)，请告诉用户：“抱歉，我暂时没有找到关于您问题的相关信息。”
        3.  综合信息进行回答，不要仅仅罗列数据。如果信息是列表，请以项目符号或自然段落的形式呈现。
        4.  不要在回答中提及 "Cypher"、"查询"、"数据库"、"JSON" 等技术术语。
        5.  直接呈现最终答案，不要说“根据检索到的信息...”。
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.str_parser

        return chain.invoke({
            "question": question,
            "cypher_query": cypher_query,
            "query_result": processed_result_str
        })

    def chat_enhanced(self, question: str):
        print(f"\n[User Question]: {question}")
        start = time.time()
        cypher_data = self._generate_cypher_enhanced(question)
        print(f"cypher生成耗时：{time.time() - start}")
        if not cypher_data or "cypher_query" not in cypher_data:
            return "抱歉，我无法理解您的问题来生成一个有效的查询。"
        print(cypher_data)
        cypher = cypher_data['cypher_query']
        entities_to_align = cypher_data['entities_to_align']
        print(f"\n[Generated Cypher]:\n{cypher}")
        print(f"[Entities to Align]: {entities_to_align}")

        if entities_to_align:
            start = time.time()
            aligned_entities = self._entity_align_enhanced(entities_to_align, question)
            print(f"检索对齐：{time.time() - start}")
            if not aligned_entities:
                entity_names = ", ".join([f"'{e['entity']}'" for e in entities_to_align])
                return f"抱歉，我在知识库中找不到与 {entity_names} 相关确切信息，请您换个问法试试。"
            print(f"[Aligned Entities]: {aligned_entities}")
        else:
            query_result = self.graph.query(cypher)
            start = time.time()
            answer = self._generate_answer_enhanced(question, cypher, query_result)
            print(f"最终生成耗时：{time.time() - start}")
            print(f"\n[Final Answer]: {answer}")
            return answer
        try:
            params = {item['param_name']: item['entity'] for item in aligned_entities}
            query_result = self.graph.query(cypher, params=params)
            print(f"\n[Raw Query Result]:\n{json.dumps(query_result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"Error executing Cypher query: {e}")
            return "抱歉，我在查询知识库时遇到了一个问题，暂时无法回答您。"
        start = time.time()
        answer = self._generate_answer_enhanced(question, cypher, query_result)
        print(f"最终生成耗时：{time.time() - start}")
        print(f"\n[Final Answer]: {answer}")
        return answer
    def chat(self, question):
        return self.chat_enhanced(question)


if __name__ == '__main__':
    chat_service = ChatService()
    while True:
        user_input = input('请输入问题 (输入 "exit" 退出): ')
        if user_input.lower() == 'exit':
            break
        chat_service.chat(user_input)