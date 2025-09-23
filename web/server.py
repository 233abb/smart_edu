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
            model='gemini-2.5-flash-lite',
            model_provider="google_genai",
            temperature=0.5,
            streaming=True
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
        print(self.graph.schema)

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

    # ===================================================================
    # =================== 阶段 1: 模型A - 规划师 ========================
    # ===================================================================
    async def _run_model_a_planner(self, question: str):
        prompt_template = """
        你是一位顶级的知识图谱查询规划师，具备强大的推理、纠错和意图理解能力。

        **图谱Schema信息:**
        {schema_info}
        
        **核心规则 (必须始终遵守):**
        1.  你生成的所有节点标签 (label)，其值 **必须** 来自上面的【图谱Schema信息】。
        2.  绝不允许创造Schema中不存在的任何标签。
        3.  你生成的所有查询路径中的每一段关系，**必须** 严格匹配【图谱Schema信息】中定义的方向。
        **示例**: 如果Schema定义了 `(:Course)-[:TAUGHT_BY]->(:Teacher)`，那么从课程找老师的路径是 `(c)-[:TAUGHT_BY]->(t)`。如果想从老师找课程，绝不能写成 `(t)-[:TAUGHT_BY]->(c)`，而**必须**使用反向语法 `(t)<-[:TAUGHT_BY]-(c)`。

        ---
        **核心任务:**
        你的任务是严格遵循下面的【思维链】进行思考，并最终生成一个JSON格式的规划。
        
        **【思维链 (Chain-of-Thought)】:**
        
        **步骤 1: 实体识别与归一化 (Entity Recognition & Normalization)**
            **1a. 实体存在性判断**: 首先判断问题是否在询问关于某个或某些**具体命名实体**的信息（如“小明的考试情况”）。
            - **如果问题是关于一个实体类型的全局查询**（如“谁参加了考试？”、“有哪些课程？”），则**实体列表为空是正确的**。在这种情况下，记录下没有具体实体，然后可以直接进行步骤3。
            - **如果问题包含具体实体**，则继续执行下面的步骤。
        **1b. 检查与归一化**: 检查提取的实体是否存在任何形式的错误或非标准表达。你必须处理以下情况：
        *   **a) 拼写错误**: 如 'jvav' -> 'Java'。
        *   **b) 同音/拼音输入错误**: 如 '鸡器学习' -> '机器学习'。
        *   **c) 常用缩写或不完整表达**: 如 '深学习' -> '深度学习'。
        *   你必须结合上下文、常识以及【图谱Schema信息】中的知识（如已知的课程名、知识点名），将这些模糊的输入**归一化**到图谱中对应的标准实体名称上。在后续步骤中，必须使用归一化后的实体。
        
        **步骤 2: 实体链接与标签分配 (Entity Linking & Labeling)**
        -   **2a. 分析实体上下文**: 对于每个修正后的实体 (例如 "Java")，分析它在问题中的上下文含义。
        -   **2b. 候选标签生成**: 查看【图谱Schema】，列出该实体可能对应的所有节点标签。
        -   **2c. 关系可能性剪枝**: 根据用户问题的意图 (例如 "介绍一下" 或 "有哪些章节") 和【图谱Schema】中的关系，排除不合理的标签。
        -   **2d. 最终标签确定**: 基于上述分析，为每个实体确定最可能的一个标签。
        
        **步骤 3: 意图分类与路径规划 (Intent Classification & Path Finding)**
        -   **3a. 意图判断**: 综合分析用户问题，判断其核心意图 (`graph_query`, `general_knowledge`, `unanswerable`)。这个判断必须基于图谱的实际能力（例如，是否有可供查询的属性或关系）。
        -   **3b. 查询路径构建**: 如果意图是 `graph_query`，基于已链接的实体和意图，设计清晰的图查询路径伪代码。**必须严格遵守【图谱Schema信息】中定义的关系方向**。例如，Schema 定义了 `(:Course)-[:TAUGHT_BY]->(:Teacher)`，那么从课程找老师的路径是合法的；但反向路径 `(:Teacher)-[:TAUGHT_BY]->(:Course)` 是非法的，必须使用 `(:Teacher)<-[:TAUGHT_BY]-(:Course)`。
        
        **步骤 4: 问题重写与格式化输出 (Question Rewriting & Final Formatting)**
        -   **4a. 重写问题**: 根据最终确定的意图和路径，生成一个或多个清晰、无歧义的 `rewritten_questions`。如果查询是概览性的，可以生成多个问题；否则只生成一个。非 `graph_query` 类型此列表为空。
        -   **4b. 格式化输出**: 将以上所有分析结果，严格按照下面【最终输出格式】的要求进行组织。`reasoning` 字段必须包含你完整的、未简化的思考过程。
        
        ---
        **最终输出格式 (Final Output Format):**
        必须严格使用以下JSON格式输出，不要包含任何额外的解释或注释。
        ```json
        {{
          "query_type": "...",
          "analysis": {{
            "reasoning": "步骤1: ...\\n步骤2: ...\\n步骤3: ...",
            "intent": "...",
            "entities": [
              {{
                "original_text": "...", // 可选
                "name": "...",
                "label": "..."
              }}
            ],
            "query_path_logic": "..."
          }},
          "rewritten_questions": []
        }}
        ---
        **示例 1 (概览性问题):**
        *   用户问题: "介绍一下小明的学习情况"
        *   输出:
            ```json
            {{
              "query_type": "graph_query",
              "analysis": {{
                "reasoning": "步骤1: 识别与纠错。从问题中识别出实体 '小明'。经检查，'小明' 是一个常见的名字，拼写正确，无需修正。\\n步骤2: 实体链接。分析上下文 '学习情况'，这强烈暗示 '小明' 是一个学习者角色。在Schema中查找可能的标签，`User` 和 `Teacher` 都是人。但描述学习行为的关系如 `FAVORS`, `TOOK_EXAM`, `WATCHED` 都明确地从 `User` 节点出发。因此，将 '小明' 精确链接到 `User` 标签。\\n步骤3: 意图分类与路径规划。'学习情况' 是一个开放式、概览性的意图，需要查询多个方面的信息。这属于 `graph_query`。为了全面回答，需要规划一个并行的多路径查询：从(u:User)节点出发，分别查找其收藏的课程、参加的考试和观看过的视频。",
                "intent": "获取用户“小明”在平台上的所有学习活动概览。",
                "entities": [
                  {{"name": "小明", "label": "User"}}
                ],
                "query_path_logic": "从指定的 (u:User) 节点出发，并行查询三条路径：\\n1. (u)-[:FAVORS]->(c:Course)\\n2. (u)-[:TOOK_EXAM]->(p:Paper)\\n3. (u)-[:WATCHED]->(v:Video)"
              }},
              "rewritten_questions": [
                "查询用户“小明”收藏了哪些课程？",
                "查询用户“小明”参加过哪些考试？",
                "查询用户“小明”看过的视频有哪些？"
              ]
            }}
            ```
        ---
        **示例 2 (多跳查询):**
        *   用户问题: "小明收藏的《高等数学》这门课是哪个老师教的？"
        *   输出:
            ```json
            {{
              "query_type": "graph_query",
              "analysis": {{
                "reasoning": "步骤1: 识别与纠错。识别出实体 '小明' 和 '高等数学'。两者拼写均正确。\\n步骤2: 实体链接。'小明' 在 '收藏' 的动作中是主语，链接到 `User` 标签。'高等数学' 被明确描述为 '这门课'，链接到 `Course` 标签。\\n步骤3: 意图分类与路径规划。问题的意图是查找一个通过中间节点连接的目标（老师）。这是一个典型的多跳查询，属于 `graph_query`。查询路径必须串联起来：首先从 (User '小明') 通过 `FAVORS` 关系找到 (Course '高等数学')，然后再从这个Course节点通过 `TAUGHT_BY` 关系找到最终的 (Teacher) 节点。",
                "intent": "查询特定用户收藏的特定课程的授课老师。",
                "entities": [
                  {{"name": "小明", "label": "User"}},
                  {{"name": "高等数学", "label": "Course"}}
                ],
                "query_path_logic": "执行一个两跳的串行查询：\\n(u:User {{name:'小明'}})-[:FAVORS]->(c:Course {{name:'高等数学'}})-[:TAUGHT_BY]->(t:Teacher)"
              }},
              "rewritten_questions": [
                "查询用户'小明'收藏的课程'高等数学'的授课老师是谁？"
              ]
            }}
            ```
        ---
        **示例 3 (聚合查询):**
        *   用户问题: "统计一下每个老师都教了多少门课"
        *   输出:
            ```json
            {{
              "query_type": "graph_query",
              "analysis": {{
                "reasoning": "步骤1: 识别与纠错。问题中没有特定的实体实例（如某个老师的名字），而是关于实体类型 '老师' 和 '课'。\\n步骤2: 实体链接。不适用，因为是全局查询。\\n步骤3: 意图分类与路径规划。关键词 '统计' 和 '每个' 表明这是一个全局聚合查询，属于 `graph_query`。规划路径为：首先匹配所有的 `Teacher` 节点；然后，对于每个老师，需要找到他们教的课程。根据Schema `(:Course)-[:TAUGHT_BY]->(:Teacher)`，必须使用反向关系来查找课程；最后，对每个老师关联的课程进行 `count` 聚合。",
                "intent": "按老师分组，统计每位老师教授的课程数量。",
                "entities": [],
                "query_path_logic": "这是一个全局聚合查询，不从特定实体出发：\\n1. 匹配所有 (t:Teacher) 节点。\\n2. 对于每个老师，通过反向关系 (t)<-[:TAUGHT_BY]-(c:Course) 找到其教授的课程。\\n3. 对每个老师的课程进行计数 (count)。"
              }},
              "rewritten_questions": [
                "查询所有老师及其教授的课程数量。"
              ]
            }}
            ```
        ---
        **示例 4 (需要反向关系的单跳查询):**
        *   用户问题: "《Java从入门到精通》这门课有哪些章节？"
        *   输出:
            ```json
            {{
              "query_type": "graph_query",
              "analysis": {{
                "reasoning": "步骤1: 识别与纠错。识别实体 'Java从入门到精通'，拼写正确。\\n步骤2: 实体链接。上下文 '这门课' 明确地将实体链接到 `Course` 标签。\\n步骤3: 意图分类与路径规划。意图是查找与给定课程直接关联的所有 '章节'。这是一个单跳关系查询，属于 `graph_query`。检查Schema，关系是 `(:Chapter)-[:PART_OF]->(:Course)`。因为查询的起点是Course，而目标是Chapter，所以必须沿 `PART_OF` 关系进行反向遍历才能找到答案。",
                "intent": "查询特定课程包含的所有章节。",
                "entities": [
                  {{"name": "Java从入门到精通", "label": "Course"}}
                ],
                "query_path_logic": "从指定的 (c:Course) 节点出发，通过反向关系查询单跳邻居：\\n(chap:Chapter)<-[:PART_OF]-(c)"
              }},
              "rewritten_questions": [
                "查询课程“Java从入门到精通”的所有章节。"
              ]
            }}
            ```
        ---
        **现在，开始你的任务:**
        用户问题: "{question}"
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.json_parser
        return await chain.ainvoke({"question": question,"schema_info":self.graph.schema})

    # ===================================================================
    # =================== 阶段 2: 模型B - 执行者 ========================
    # ===================================================================
    async def _run_model_b_executor(self, planner_output: str):
        # 少样本示例
        examples = """
        ### 示例 1: 全局查询
        **规划师的输入:**
        ```json
        {{
            "query_type": "graph_query",
            "analysis": {{
                "intent": "获取系统中所有的老师。",
                "entities": [],
                "query_path_logic": "这是一个全局性的实体类型检索。需要直接匹配所有类型为'Teacher'的节点。"
            }},
            "rewritten_questions": ["有哪些老师", "列出所有老师"]
        }}
        ```
        **你的输出:**
        ```json
        {{
            "cypher_query": "MATCH (t:Teacher) RETURN collect(DISTINCT t.name) AS allTeachers",
            "entities_to_align": []
        }}
        ```
    
        ### 示例 2: 单实体单关系查询
        **规划师的输入:**
        ```json
        {{
            "query_type": "graph_query",
            "analysis": {{
                "intent": "查询特定课程的所有章节。",
                "entities": [{{"name": "Java从入门到精通", "label": "Course"}}],
                "query_path_logic": "从指定的Course节点出发，通过'PART_OF'反向关系找到所有关联的Chapter节点。"
            }},
            "rewritten_questions": ["“Java从入门到精通”这门课有哪些章节？"]
        }}
        ```
        **你的输出:**
        ```json
        {{
            "cypher_query": "MATCH (c:Course {{name: $param_0}}) WITH c OPTIONAL MATCH (chap:Chapter)-[:PART_OF]->(c) RETURN c AS primaryEntity, collect(DISTINCT chap.name) AS chaptersInCourse",
            "entities_to_align": [{{"param_name": "param_0", "entity": "Java从入门到精通", "label": "Course"}}]
        }}
        ```
    
        ### 示例 3: 概览性组合查询
        **规划师的输入:**
        ```json
        {{
            "query_type": "graph_query",
            "analysis": {{
                "intent": "获取用户“小明”在平台上的所有学习活动概览。",
                "entities": [{{"name": "小明", "label": "User"}}],
                "query_path_logic": "这是一个概览性问题，需要将三个独立的查询路径组合起来：(User)-[:FAVORS]->(Course), (User)-[:TOOK_EXAM]->(Paper), 和 (User)-[:WATCHED]->(Video)。"
            }},
            "rewritten_questions": [
                "查询用户“小明”收藏了哪些课程",
                "查询用户“小明”参加过哪些考试",
                "查询用户“小明”看过的视频有哪些"
            ]
        }}
        ```
        **你的输出:**
        ```json
        {{
            "cypher_query": "MATCH (u:User {{name: $param_0}}) WITH u OPTIONAL MATCH (u)-[:FAVORS]->(c:Course) OPTIONAL MATCH (u)-[:TOOK_EXAM]->(p:Paper) OPTIONAL MATCH (u)-[:WATCHED]->(v:Video) RETURN u AS primaryEntity, collect(DISTINCT c.name) AS favoritedCourses, collect(DISTINCT p.title) AS takenExams, collect(DISTINCT v.name) AS watchedVideos",
            "entities_to_align": [{{"param_name": "param_0", "entity": "小明", "label": "User"}}]
        }}
        ```
        """
        prompt_template = """
        你是一位顶级的 Neo4j Cypher 查询生成专家。你的任务是接收一个来自“规划师”模型的结构化JSON输入，并根据其中的分析和图谱Schema信息，生成一条**单一、高效、组合式**的参数化Cypher查询及相关的元数据。
        
        **图谱Schema信息:**
        {schema_info}
        
        **少样本示例:**
        {examples}
        
        **任务:**
        现在，请根据下面的规划师输入生成Cypher查询和需要对齐的实体。
        **规划师的输入:**
        ```json
        {planner_input}
        ```
        **核心要求:**
        1.  **输入是结构化的**: 你的输入是一个JSON对象，其中 `analysis` 字段包含了意图、已识别的实体和查询路径逻辑。
        2.  **直接利用实体信息**: **必须**使用输入中 `analysis.entities` 数组的信息来生成输出的 `entities_to_align` 列表。不要重新识别实体。
        3.  **遵循查询逻辑**: `analysis.query_path_logic` 提供了构建查询的核心思路。结合 `rewritten_questions` 来理解需要聚合返回哪些具体数据。
        4.  **必须严格遵守【图谱Schema信息】中定义的关系方向**。例如，Schema 定义了 `(:Course)-[:TAUGHT_BY]->(:Teacher)`，那么从课程找老师的路径是合法的；但反向路径 `(:Teacher)-[:TAUGHT_BY]->(:Course)` 是非法的，必须使用 `(:Teacher)<-[:TAUGHT_BY]-(:Course)`。
        5.  **技术规范**:
            *   使用 `OPTIONAL MATCH` 来安全地连接不同的查询路径。
            *   使用 `collect(DISTINCT ...)` 函数来聚合每个路径的结果。
            *   为每个聚合结果使用清晰的别名 (e.g., `AS collectedCourses`)。
            *   生成参数化的Cypher查询，参数以 `$` 开头 (e.g., `$param_0`)。
        
        
        **要求:**
        严格使用以下JSON格式输出结果，不要包含任何额外的解释或注释。
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
            planner_input_str = json.dumps(planner_output, ensure_ascii=False, indent=2)
            print(planner_input_str)
            return await chain.ainvoke({
                "planner_input": planner_input_str,
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

    async def _generate_answer_enhanced(self, question: str,rewrite_question,processed_result: dict):
        """
        Generates a natural language answer using the question, the query, and preprocessed results.
        """
        prompt_template = """
        你是一个智能教学助手。请根据用户问题、用于查询知识图谱的Cypher语句以及查询结果，生成一个友好、自然的中文回答。

        ### 用户问题:
        {question}
        
        ### 重写问题:
        {rewrite_question}
        
        ### 检索到的信息 (JSON格式):
        {processed_result}

        ### 回答要求:
        1.  用流畅、对话式的中文进行回答，直接回答用户的问题。
        2.  如果检索到的信息为空 (`[]`)，请告诉用户：“抱歉，我暂时没有找到关于您问题的相关信息。”
        3.  综合信息进行回答，不要仅仅罗列数据。如果信息是列表，请以项目符号或自然段落的形式呈现。
        4.  在列举完所有信息后，进行一个简短的总结性陈述，比如“以上就是参加考试的同学和他们的成绩情况。”或者“希望这些信息对您有帮助！”
        4.  不要在回答中提及 "Cypher"、"查询"、"数据库"、"JSON" 等技术术语。
        5.  直接呈现最终答案，不要说“根据检索到的信息...”。
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.str_parser

        async for chunk in chain.astream({
            "question": question,
            "processed_result": json.dumps(processed_result, ensure_ascii=False, indent=2,default=str),
            "rewrite_question":json.dumps(rewrite_question, ensure_ascii=False, indent=2,default=str)
        }):
            yield chunk

    async def stream_chat_pipeline(self, question: str):
        print(f"\n[User Question]: {question}")
        # --- 步骤 1: 模型A规划 ---
        start = time.time()
        yield {"type": "status", "message": "分析您的问题..."}
        model_a_output = await self._run_model_a_planner(question)
        print(f"cypher生成耗时：{time.time() - start}")
        # --- 步骤 2: 模型B执行 ---
        start = time.time()
        yield {"type": "status", "message": "生成图数据库查询..."}
        model_b_output = await self._run_model_b_executor(model_a_output)
        print(model_b_output)
        if not model_b_output or "cypher_query" not in model_b_output:
            yield {"type": "error", "message": "抱歉，我无法理解您的问题来生成一个有效的查询。"}
        cypher = model_b_output['cypher_query']
        entities_to_align = model_b_output['entities_to_align']
        yield {"type": "cypher", "query": cypher}  # 可用于调试
        print(f"cypher生成耗时：{time.time() - start}")

        print(cypher)
        print(f"\n[Generated Cypher]:\n{cypher}")
        print(f"[Entities to Align]: {entities_to_align}")

        # --- 步骤 3: 实体对齐 ---
        if entities_to_align:
            yield {"type": "status", "message": "校准知识库实体..."}
            aligned_entities = self._entity_align_enhanced(entities_to_align,question)
            print(aligned_entities)
            if not aligned_entities:
                entity_names = ", ".join([f"'{e['entity']}'" for e in entities_to_align])
                yield {"type": "error",
                       "message": f"抱歉，我在知识库中找不到与 {entity_names} 相关确切信息，请您换个问法试试。"}
                return
            params = {item['param_name']: item['entity'] for item in aligned_entities}
        else:
            params = {}
        # --- 步骤 4: 数据库查询 ---
        yield {"type": "status", "message": "正在查询知识图谱..."}
        print(cypher)
        query_result = self.graph.query(cypher, params=params)
        print(query_result)

        # --- 步骤 6: 模型C播报 (流式) ---
        yield {"type": "status", "message": "生成最终回答..."}
        async for chunk in self._generate_answer_enhanced(question,model_a_output,query_result):
            yield {"type": "chunk", "content": chunk}
        yield {"type": "final", "message": "回答完毕"}


if __name__ == '__main__':
    chat_service = ChatService()
    while True:
        user_input = input('请输入问题 (输入 "exit" 退出): ')
        if user_input.lower() == 'exit':
            break
        chat_service.chat(user_input)