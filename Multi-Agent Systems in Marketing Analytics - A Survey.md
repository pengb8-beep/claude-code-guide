# Multi-Agent Systems in Marketing Analytics: From Agent-Based Modeling to LLM-Driven Collaborative Intelligence

## Abstract

Multi-Agent Systems (MAS), a core paradigm of distributed artificial intelligence, have undergone a profound evolution in marketing analytics—from traditional simulation modeling to large language model (LLM)-driven collaborative intelligence. In the early stages, Agent-Based Modeling (ABM) provided "bottom-up" analytical tools for complex marketing phenomena such as market diffusion, word-of-mouth propagation, and competitive dynamics by simulating individual consumer decision-making and interaction processes. In recent years, the emergence of LLM-based multi-agent collaboration systems (e.g., AutoGen, CrewAI, MetaGPT) has opened an entirely new paradigm in which AI agents autonomously collaborate to execute marketing tasks—from market research and content generation to advertising optimization, multiple AI agents with specialized roles can achieve automated orchestration of complex marketing workflows through dialogue, reasoning, and tool invocation. This paper systematically reviews the research trajectory of multi-agent systems in marketing analytics, covering theoretical foundations, traditional ABM applications, LLM multi-agent architectures, core application scenarios, evaluation methods, and key challenges. It aims to provide researchers with a comprehensive knowledge landscape spanning two generations of technological paradigms and to outline future directions for multi-agent marketing intelligence.

**Keywords:** Multi-Agent Systems; Agent-Based Modeling; Large Language Models; Marketing Analytics; Collaborative Intelligence; Consumer Behavior Simulation

---

## 1 Introduction

The core objective of marketing analytics is to understand consumer behavior, evaluate the effectiveness of marketing strategies, and optimize marketing decisions. However, real-world marketing environments are inherently complex adaptive systems composed of large numbers of heterogeneous individuals (consumers, firms, intermediary platforms, etc.) interacting with one another. Micro-level individual decisions aggregate into macro-level market phenomena through social networks, market mechanisms, and information dissemination channels (Rand & Rust, 2011). Traditional marketing analytics methods—whether econometric models or machine learning approaches—typically treat consumers as homogeneous or limitedly heterogeneous groups, making it difficult to fully capture the effects of inter-individual interactions and emergence on market dynamics (Bonabeau, 2002). As Farmer and Foley (2009) advocated in *Nature*, the analysis of economic and market systems urgently requires agent-based modeling approaches to complement the limitations of traditional methods.

Multi-Agent Systems (MAS) provide a unique methodological perspective for understanding and analyzing such complex marketing phenomena. Within the MAS framework, each agent is a computational entity with autonomous decision-making capabilities, able to perceive its environment, interact with other agents, and take actions based on its own objectives (Wooldridge, 2009). When large numbers of agents interact in a shared environment, complex patterns emerge at the system level that cannot be directly predicted from individual behaviors—as Schelling (1971) revealed in his classic segregation model, slight preference differences at the individual level can produce extreme spatial separation patterns at the group level. This micro-to-macro emergence mechanism closely mirrors the operational logic of real markets.

The application of multi-agent approaches in marketing analytics dates back to the late 20th century. Epstein and Axtell's (1996) seminal work *Growing Artificial Societies* laid the theoretical foundation for ABM, demonstrating how complex social phenomena can emerge from simple individual interaction rules. Subsequently, ABM has been widely applied in market diffusion modeling (Goldenberg et al., 2001), consumer choice behavior simulation (Zhang & Chang, 2012), and competitive strategy analysis (Garcia, 2005). Macy and Willer (2002) noted in the *Annual Review of Sociology* that ABM marks a methodological shift in social science from "factor analysis" to "actor analysis"—a perspective equally applicable to marketing research. Rand and Rust (2011), in their review published in the *International Journal of Research in Marketing*, explicitly argued that ABM should become an essential component of the marketing researcher's toolkit, highlighting its unique advantages in handling heterogeneity, social network effects, and nonlinear dynamics. Negahban and Yilmaz (2014) further provided an integrative review of ABM applications in marketing research, identifying the field's major research themes and methodological trends.

In recent years, breakthroughs in large language models (LLMs) have given rise to an entirely new multi-agent paradigm. Unlike agents in traditional ABM that operate on predefined rules or simple heuristics, LLM-based agents possess powerful natural language understanding and generation capabilities, complex reasoning abilities, and tool-use competencies (Brown et al., 2020; OpenAI, 2023). The ReAct framework proposed by Yao et al. (2023) demonstrated how LLM agents can combine reasoning with acting, providing a technical foundation for autonomous agent decision-making and environmental interaction. Park et al.'s (2023) "Generative Agents" research showed that LLM agents can produce credible human behavior simulations, opening new possibilities for consumer behavior simulation. Wu et al.'s (2023) AutoGen framework provided a general technical infrastructure for multi-agent dialogue and collaboration.

However, the academic community currently lacks a systematic review that bridges the research lineages of traditional ABM and emerging LLM multi-agent systems, comprehensively mapping the application landscape of multi-agent methods in marketing analytics. This paper aims to fill this gap. Our main contributions include: (1) constructing a technological evolution framework from traditional ABM to LLM multi-agent systems; (2) systematically reviewing the current state of multi-agent system applications across core marketing analytics domains; (3) comparatively analyzing the strengths and limitations of both technological paradigms (see Table 2); and (4) proposing future research directions and open questions.

The remainder of the paper is organized as follows: Section 2 introduces the literature review methodology; Section 3 provides an overview of the theoretical foundations and technological evolution of multi-agent systems; Sections 4 through 7 discuss multi-agent applications in consumer behavior simulation, market competition and diffusion modeling, marketing task automation and collaboration, and personalized marketing and recommendation, respectively; Section 8 discusses evaluation methods and benchmarks; Section 9 analyzes key challenges and ethical issues; Section 10 outlines future research directions; and Section 11 concludes the paper.

---

## 2 Literature Review Methodology

To ensure the systematicity and comprehensiveness of this review, a structured literature search and screening strategy was employed.

**Search Strategy**: Five academic databases were searched: Web of Science, Scopus, Google Scholar, ACM Digital Library, and IEEE Xplore. Search keywords included two sets of combinations: (1) Traditional ABM: "agent-based model*" AND ("marketing" OR "consumer behavior" OR "market simulation" OR "diffusion"); (2) LLM multi-agent: "multi-agent" AND ("large language model" OR "LLM" OR "GPT") AND ("marketing" OR "advertising" OR "analytics"), as well as "AI agent" AND "marketing". The search time span was from 1996 to 2025, covering the complete trajectory from early ABM applications to the latest LLM multi-agent advances.

**Screening Criteria**: The initial search yielded approximately 400 articles, which were narrowed to 75 core references using the following criteria: (1) published in peer-reviewed journals or top-tier academic conferences, or as preprints with significant impact; (2) explicitly involving multi-agent methods in marketing-related theoretical, methodological, or applied contexts; (3) priority given to highly cited works and representative contributions in each sub-area; (4) balanced coverage across marketing science, computer science, and complex systems science perspectives.

**Literature Distribution**: Temporally, traditional ABM literature is concentrated mainly between 2001 and 2018, while LLM multi-agent literature has been published almost entirely after 2023, reflecting the rapid emergence of this field. Thematically, consumer behavior simulation and market diffusion are the core applications of traditional ABM, while marketing task automation and content generation are the primary focus of LLM multi-agent research.

---

## 3 Theoretical Foundations and Technological Evolution

### 3.1 Theoretical Foundations of Multi-Agent Systems

The theoretical roots of multi-agent systems are grounded in distributed artificial intelligence and complex adaptive systems theory. A multi-agent system consists of multiple agents and their shared environment, with each agent possessing the following core characteristics: (1) **Autonomy**—the ability to make independent decisions without direct external control; (2) **Sociability**—the ability to interact with other agents through communication protocols; (3) **Reactivity**—the ability to perceive and respond to environmental changes; and (4) **Proactiveness**—the ability to take goal-directed initiatives (Wooldridge, 2009).

In the marketing context, agents can represent consumers, firms, advertising platforms, opinion leaders, or regulatory bodies. Interaction mechanisms among agents include direct communication (e.g., word-of-mouth between consumers), indirect communication (e.g., indirect coordination through market price signals), and environmental influence (e.g., competition for consumer attention through advertising). These interactions collectively drive the emergence of macro-level patterns at the market level (Tesfatsion, 2006).

Complex Adaptive Systems (CAS) theory provides a key conceptual framework for understanding market dynamics. Holland (1995) identified the core features of CAS as emergence, self-organization, adaptation, and nonlinear feedback. Marketing environments fit these characteristics perfectly: the emergence of consumer trends, the self-organization of brand communities, firms' adaptation to market signals, and the nonlinear cascade effects of word-of-mouth propagation—particularly the "small-world network" properties revealed by Watts and Strogatz (1998), which enable information and influence to propagate through consumer networks at speeds far exceeding expectations. As the primary computational tool for studying CAS, ABM can generate system-level emergent phenomena "bottom-up" from individual behavioral rules, something that traditional "top-down" econometric approaches find difficult to achieve (Bonabeau, 2002).

### 3.2 Technological Evolution: From Rule-Driven to Cognition-Driven

The technological evolution of multi-agent systems in marketing analytics can be divided into three phases, as shown in Table 1.

**Table 1: Technological Evolution of Multi-Agent Marketing Analytics**

| Phase | Period | Agent Type | Decision Mechanism | Representative Applications | Limitations |
|-------|--------|-----------|-------------------|---------------------------|-------------|
| Rule-Driven ABM | 1996–2010 | Simple reactive agents | If-then rules, threshold models | Diffusion simulation, market entry | Oversimplified behavioral rules |
| Learning ABM | 2010–2022 | Adaptive agents | Reinforcement learning, evolutionary algorithms, Bayesian updating | Dynamic pricing games, competitive strategy | Lack of language and reasoning capabilities |
| LLM Multi-Agent | 2023–Present | Cognitive agents | LLM reasoning, tool invocation, memory retrieval | Marketing task automation, consumer simulation | High cost, hallucination, controllability |

**Phase 1: Rule-Driven ABM (1996–2010)**. Early ABM agents operated on predefined decision rules. For example, in classical innovation diffusion models, consumer agents decided whether to adopt a new product based on whether the proportion of adopted neighbors in their social network exceeded a threshold (Goldenberg et al., 2001). Gilbert's (2008) monograph systematically articulated the methodological foundations of ABM during this period. This phase provided intuitive computational tools for understanding market emergence, but the overly simplistic behavioral rules of agents failed to reflect the cognitive complexity of real consumers.

**Phase 2: Learning ABM (2010–2022)**. Researchers introduced machine learning methods such as reinforcement learning, evolutionary computation, and Bayesian inference into ABM agents, enabling them to learn and adapt from experience. For example, Multi-Agent Reinforcement Learning (MARL) was applied to research on dynamic pricing competition and advertising bidding strategies (Tesauro & Kephart, 2002; Jin et al., 2018). Dawid and Delli Gatti (2018), in their review of agent-based macroeconomic modeling, noted that learning agents demonstrated unique value in simulating the dynamic formation of market equilibria. Agents in this phase gained certain learning and adaptation capabilities but still lacked the ability to process natural language, commonsense reasoning, and creative tasks.

**Phase 3: LLM Multi-Agent (2023–Present)**. The advent of large language models endowed agents with human-like language comprehension, reasoning, and generation capabilities. LLM agents can not only process structured data but also understand and generate natural language instructions, use external tools (e.g., search engines, data analysis APIs), and coordinate complex tasks with other agents through dialogue (Xi et al., 2023). This marks a paradigm shift in multi-agent systems from "simulating the world" (ABM) to "acting in the world" (AI Agent). Chen et al.'s (2023) research published in *PNAS* further demonstrated that GPT models can exhibit economically rational behavior, providing important theoretical support for the application of LLM agents in market simulation.

### 3.3 Architecture of LLM Multi-Agent Systems

Current mainstream LLM multi-agent systems typically comprise the following core components (Wang et al., 2024):

**Agent Architecture**: Each LLM agent generally consists of four modules: (1) **Core LLM**—providing reasoning and language capabilities (e.g., GPT-4, Claude, LLaMA; see Touvron et al., 2023); (2) **Role Definition** (System Prompt)—defining the agent's professional identity, knowledge scope, and behavioral constraints; (3) **Memory System**—including short-term working memory (dialogue context) and long-term memory (experiences and knowledge stored in vector databases); and (4) **Tool Interface**—allowing agents to invoke external tools for specific tasks (e.g., data queries, code execution, web browsing). The ReAct framework (Yao et al., 2023) provides an effective prompting paradigm for the agent's reasoning-action loop, enabling LLMs to alternate between thinking and tool invocation.

**Collaboration Mechanisms**: Collaboration patterns among multiple agents include (Guo et al., 2024): (1) **Sequential Pipeline**—agents process tasks in a predefined sequential order; (2) **Debate and Discussion**—multiple agents present different viewpoints on the same issue and reach consensus through dialogue; (3) **Hierarchical Management**—manager agents handle task decomposition and assignment while worker agents execute specific subtasks; and (4) **Dynamic Orchestration**—adaptively adjusting agent division of labor and collaboration workflows based on task state.

**Representative Frameworks**: AutoGen (Wu et al., 2023) provides flexible multi-agent dialogue orchestration capabilities; CrewAI (Moura, 2024) emphasizes role-based team collaboration; MetaGPT (Hong et al., 2024) draws on standardized software engineering processes to organize multi-agent collaboration. The Chameleon framework proposed by Lu et al. (2024) demonstrates how to build multi-agent systems with compositional reasoning capabilities through plug-and-play module combinations.

### 3.4 ABM Simulation Tools and Platforms

The development of ABM has been inseparable from dedicated simulation tools. Abar et al. (2017) provided a comprehensive review of state-of-the-art ABM simulation software. Major tools include:

- **NetLogo**: The most popular ABM tool for teaching and research, offering intuitive visualization and a rich model library (Wilensky & Rand, 2015). Its built-in social network generators and GIS extensions make it particularly suitable for marketing diffusion research.
- **Repast**: A Java-based platform for large-scale simulation, supporting distributed parallel computing, suitable for market simulations with millions of agents.
- **Mesa**: A Python-based ABM framework that integrates easily with the machine learning and data analytics ecosystem, with rapidly growing adoption among marketing researchers in recent years.
- **AnyLogic**: A commercial-grade multi-method simulation platform supporting hybrid modeling with ABM, system dynamics, and discrete event simulation, widely used in industry marketing simulations.

Macal and North (2010) noted in their ABM tutorial that tool selection should be based on a comprehensive consideration of the research problem's scale, agent complexity, and integration requirements with other analytical methods. For LLM multi-agent systems, the primary tools are currently frameworks such as AutoGen and CrewAI; the integration of these tools with traditional ABM platforms represents an area for future development.

Building on this overview of the technical foundations, the following sections examine specific applications of multi-agent methods across core marketing analytics domains, beginning with the most established area: consumer behavior simulation.

---

## 4 Consumer Behavior Simulation

Consumer behavior simulation is the most established application domain for multi-agent methods in marketing analytics. From ABM's rule-driven simulation to LLM-powered cognitive simulation, multi-agent methods have provided increasingly sophisticated modeling tools for understanding consumer decision-making.

### 4.1 ABM-Based Consumer Decision Modeling

The core value of ABM in consumer behavior research lies in its ability to model the effects of individual heterogeneity and social interactions on macro-level market outcomes. An (2012), in a review of ABM-based human decision modeling, noted that ABM's unique advantage lies in its ability to integrate cognitive theories, social networks, and spatial environments to simulate decision processes.

In **brand choice** research, ABM has been used to simulate the dynamic choice processes of consumers in multi-brand competitive environments. Zhang and Chang (2012) constructed an ABM incorporating consumer social networks and brand loyalty evolution, revealing the nonlinear relationship between word-of-mouth effects and brand switching behavior. Compared with traditional discrete choice models, ABM can capture the dynamic process of consumer preferences changing with social influence, rather than assuming static preferences.

In **purchase decision path** research, ABM has been used to simulate the complete consumer decision process from need recognition and information search to final purchase. Delre et al.'s (2007) research used ABM to demonstrate how social network structure affects consumers' information acquisition and decision speed, finding that small-world network structures are more conducive to rapid new product information dissemination than random networks—a finding that echoes the small-world network properties theorized by Watts and Strogatz (1998).

**Heterogeneity modeling** is a key advantage of ABM. Each consumer agent can be endowed with different demographic characteristics, preference parameters, social network positions, and decision strategies, producing richer behavioral diversity than representative consumer models. Hamill and Gilbert (2009) systematically discussed how to construct agent populations with socio-demographic heterogeneity in ABM, providing a methodological foundation for consumer simulation in marketing research.

### 4.2 LLM-Based Consumer Simulation

The advent of LLMs has brought a paradigm-level breakthrough to consumer behavior simulation. Traditional ABM requires researchers to pre-specify consumer decision rules, while LLM agents can autonomously generate more natural and complex consumer behaviors based on large-scale learning of human behavior.

Park et al.'s (2023) "Generative Agents" research, presented at UIST 2023, is a milestone in this direction. The researchers created 25 LLM-powered agents, each with a unique identity, memory, and social relationships, autonomously living and interacting in a virtual town. Results showed that these agents exhibited credible social behaviors, including information dissemination, relationship formation, and collective behavior—precisely the social dynamic processes central to marketing research.

In marketing-specific scenarios, LLM consumer simulation has begun to demonstrate application value. Brand et al. (2023) investigated whether LLMs can reproduce the results of classic consumer behavior experiments, finding that GPT-4 showed high consistency with human subjects in reproducing several decision biases (e.g., framing effects, anchoring effects), but exhibited systematic biases in certain risk decision scenarios. Argyle et al. (2023) proposed the concept of "Algorithmic Fidelity," examining the extent to which LLMs can faithfully simulate the attitudes and behaviors of different demographic groups—a question of fundamental importance for the reliability of LLM consumer simulation.

Horton (2023) conceptualized LLMs as "silicon subjects" (Homo Silicus), systematically exploring the possibilities and limitations of using LLMs to replace or supplement human subjects in economics and marketing experiments. The research found that LLMs can produce experimental results similar to humans under specific conditions, providing a new methodological tool for rapid prototyping and hypothesis testing in market research. Xing et al. (2024) further applied LLM consumer simulation to digital advertising scenarios, simulating consumer responses to different ad creatives and placement strategies to provide low-cost pre-evaluation for advertising optimization.

### 4.3 Multi-Agent Consumer Social Simulation

Extending consumer behavior simulation to the multi-agent social simulation level enables the study of emergent phenomena among consumer groups.

In **word-of-mouth simulation**, multi-agent models have been used to study the propagation dynamics of positive and negative word-of-mouth in social networks. Goldenberg et al.'s (2001) classic ABM study showed that a small number of highly connected consumers have a disproportionate influence on word-of-mouth propagation speed. Zhang et al. (2017) further used ABM to study viral marketing propagation mechanisms, finding complex interaction effects among network structure, seed node selection, and content attractiveness. In recent years, LLM multi-agent systems have enabled consumers in word-of-mouth simulations to generate natural language product reviews and recommendation texts, rather than simple binary signals, thereby more realistically simulating the online word-of-mouth ecosystem (Li et al., 2024).

In **group polarization and filter bubble** research, multi-agent models have been used to explore how recommendation algorithms affect consumers' information diversity and preference polarization. Flache et al. (2017) provided a comprehensive review of social influence models, revealing the conditions and mechanisms under which personalized recommendations may lead to consumer preference convergence or polarization—findings of significant value for the ethical scrutiny of platform marketing strategies. Shao et al. (2024) further proposed combining ABM with generative AI to build a new paradigm for social simulation.

Consumer behavior simulation provides the "micro-foundations" for understanding individual decisions and group dynamics, while the macro-level market phenomena that emerge from these micro-behaviors—such as innovation diffusion and competitive equilibria—are the subjects of market competition and diffusion modeling.

---

## 5 Market Competition and Diffusion Modeling

### 5.1 Multi-Agent Models of Innovation Diffusion

Innovation diffusion is the most classic domain for ABM application in marketing research. The traditional Bass diffusion model (Bass, 1969) assumes a homogeneous market, with the diffusion process driven by two parameters: the innovation coefficient and the imitation coefficient. ABM has greatly enriched the analytical framework for diffusion research by modeling individual heterogeneity and social network structure.

Goldenberg et al. (2001) proposed a cellular automata-based innovation diffusion ABM, studying for the first time at the individual level the differentiated roles of "strong ties" (close social relationships) and "weak ties" (casual acquaintances) in the diffusion process. They found that weak ties play a critical "bridge" role in the early stages of diffusion—a finding that echoes and quantifies Granovetter's (1973) "strength of weak ties" theory.

Delre et al. (2010) further incorporated consumer heterogeneity (different adoption thresholds) and social network topology into diffusion ABM, studying the effects of targeted seeding strategies on new product diffusion speed and ultimate market penetration. The research found that when consumers' adoption thresholds exhibit large heterogeneity, seeding to low-threshold "early adopters" is the optimal strategy; however, when heterogeneity is small, the seed's network position (centrality) is more important than personal characteristics.

Kiesling et al. (2012) provided a comprehensive review of ABM applications in innovation diffusion research, summarizing modeling approaches along three dimensions—agent decision rules, social network structure, and external influence factors (e.g., advertising, price)—and identified empirical calibration, i.e., how to calibrate ABM parameters with real data, as the field's core challenge.

### 5.2 Competitive Strategy and Market Dynamics

Multi-agent methods are naturally suited for modeling market competition scenarios, as competition is inherently about strategic interactions among multiple decision-makers.

Garcia (2005) used ABM to study competitive dynamics in technology innovation markets, finding that firms' innovation strategy choices (incremental vs. radical innovation) and market structure exhibit complex co-evolutionary relationships. ABM can reveal path dependence and lock-in effects in this co-evolutionary process, whereas traditional game-theoretic models typically assume market structure as exogenously given.

In the domain of **dynamic pricing competition**, Multi-Agent Reinforcement Learning (MARL) has been widely applied. Tesauro and Kephart's (2002) pioneering research demonstrated that when multiple pricing agents use Q-learning for price optimization, the system may converge to a collusive pricing equilibrium. Calvano et al. (2020), in research published in the *American Economic Review*, further confirmed this phenomenon: pricing agents independently using Q-learning algorithms can "spontaneously" learn collusive strategies, even without any explicit communication or coordination. This "algorithmic collusion" finding has attracted widespread attention from academia and regulatory bodies, posing new challenges to competition law and antitrust policy—current legislation primarily targets explicit collusion between firms rather than implicit algorithmic-level coordination.

Jin et al. (2018) introduced deep reinforcement learning into multi-agent advertising bidding scenarios, studying strategic interactions among multiple advertisers in real-time bidding (RTB) environments. When multiple intelligent advertisers simultaneously optimize their bidding strategies, system dynamics become highly complex, with the existence and convergence of Nash equilibria becoming key theoretical questions.

### 5.3 Market Evolution and Emergence

The core advantage of ABM lies in revealing "emergence"—how macro-level market phenomena spontaneously arise from micro-level individual interactions.

In **market structure evolution**, ABM has been used to study the dynamic formation processes of market concentration, brand landscapes, and industrial ecosystems. By simulating firms' entry, exit, merger, and product innovation decisions, ABM can generate power-law distributions similar to real markets (a few large firms coexisting with many small ones) and market share fluctuation patterns (Tesfatsion, 2006).

In **fashion and trend propagation**, multi-agent models have revealed how the "S-shaped" diffusion curve of consumer trends emerges from individual-level imitation and distinction motivations. Consumers are simultaneously driven by "conformity" (following trends) and "uniqueness" (being different) motivations, and the tension between these two forces generates trend rise-and-decline cycles at the group level (Rand & Rust, 2011). Chattoe-Brown (2013) further argued that ABM possesses unparalleled advantages in capturing the dynamism and heterogeneity of such social processes.

Market competition and diffusion modeling reveals the macro-level outcomes of multi-agent interactions, while at the practical level, firms are more concerned with how to leverage multi-agent systems to automate and optimize day-to-day marketing operations.

---

## 6 Marketing Task Automation and Collaboration

The most practically significant application of LLM multi-agent systems in marketing analytics is decomposing complex marketing workflows into multiple subtasks, completed collaboratively by specialized AI agents. This represents a functional shift of multi-agent systems from "simulating marketing phenomena" to "executing marketing tasks." Tian et al. (2024) conducted a systematic review of multi-agent marketing automation systems, identifying workflow orchestration, role specialization, and quality control as three core design dimensions.

### 6.1 Multi-Agent Marketing Workflow Architecture

A typical multi-agent marketing system usually contains the following role assignments (Ke et al., 2024):

A **Market Research Agent** is responsible for information collection and analysis, capable of browsing web pages, invoking search APIs, analyzing competitor data, and generating market insight reports. A **Strategy Planning Agent** develops marketing strategy frameworks based on market research results, including target audience definition, value proposition design, and channel mix planning. A **Content Creation Agent** generates diverse marketing copy, advertising creatives, and social media content according to strategic direction. A **Data Analysis Agent** handles statistical analysis, effectiveness evaluation, and attribution modeling of marketing data. A **Quality Review Agent** reviews and provides feedback on other agents' outputs, ensuring content quality and brand consistency.

Li et al. (2024) proposed a multi-agent collaboration framework for digital marketing, in which multiple LLM agents play roles including advertising planning, copywriting, placement optimization, and effectiveness analysis, collaborating through structured dialogue protocols to complete the full marketing loop from strategy formulation to effectiveness evaluation. Experiments demonstrated that multi-agent collaboration significantly outperforms single LLM agents in creative diversity and strategic comprehensiveness.

### 6.2 Market Research and Competitive Intelligence

Multi-agent systems have demonstrated significant potential in market research automation. Traditional market research workflows (questionnaire design → data collection → data analysis → insight generation) typically require weeks and the involvement of multiple professional teams, while multi-agent systems can dramatically compress this cycle.

In competitive intelligence gathering, specialized agents can autonomously browse competitor websites, analyze pricing strategies, track product updates, and monitor social media sentiment. Multiple agents collect data from different information sources, with a synthesis agent integrating fragmented information into structured competitive intelligence reports. Gao et al.'s (2024) research explored the capability boundaries of LLM agents in information retrieval and synthesis tasks, providing a technical feasibility assessment for market research automation.

In **consumer insight extraction**, LLM agents can conduct deep analysis of large-scale consumer feedback data (online reviews, customer service records, social media posts), extracting rich insights beyond traditional sentiment analysis—including unmet needs, usage scenario descriptions, competitive comparison opinions, and product improvement suggestions. Multiple agents can focus on different data sources or analytical dimensions, with a synthesis agent generating an integrated consumer insight report.

### 6.3 Content Generation and Creative Optimization

Content marketing is one of the most straightforward application domains for LLM multi-agent systems. A single LLM may exhibit creative homogenization and self-reinforcing bias in content generation, while multi-agent collaboration introduces diverse perspectives and critical feedback to enhance content quality.

**Creative Divergence and Convergence Mechanisms**: Multiple agents can generate candidate creative proposals from different angles (e.g., rational appeal vs. emotional appeal, humorous style vs. serious style), with a review agent screening and optimizing based on brand tone, target audience characteristics, and platform specifications. Wang et al.'s (2024) survey noted that multi-agent debate and reflection mechanisms can effectively reduce hallucinations and biases of single LLMs, improving output accuracy and reliability—critical for the factual accuracy and brand safety of marketing content.

**Cross-Platform Content Adaptation**: Different social media platforms (e.g., Weibo, WeChat, TikTok, Xiaohongshu [a lifestyle-sharing platform popular in China], and LinkedIn) have different content format requirements, audience characteristics, and dissemination mechanisms. Specialized platform agents can automatically adapt core marketing messages into the optimal content format for each platform, achieving a "create once, distribute everywhere" model for efficient content operations. Ke et al.'s (2024) research showed that multi-agent systems with platform expert agents significantly outperform general single-model approaches in cross-platform content adaptation quality scores.

### 6.4 Ad Placement and Budget Optimization

In the programmatic advertising domain, multi-agent methods have been used to optimize cross-channel, multi-objective advertising placement strategies.

**Multi-Objective Coordination**: Corporate advertising campaigns typically need to simultaneously optimize multiple objectives—brand awareness, user engagement, conversion rate, and ROI. Different agents can be responsible for optimizing different objectives, with a coordination agent balancing the trade-offs among objectives at the global level, avoiding local optima caused by single-objective optimization.

**Bidding Strategy Learning**: In real-time bidding (RTB) environments, MARL methods have been used to study the strategic interactions and equilibrium behavior of multiple advertisers. Cai et al. (2017) used reinforcement learning to optimize display advertising real-time bidding strategies, while when multiple advertisers simultaneously adopt similar learning algorithms, the system dynamics become a multi-agent game problem (Jin et al., 2018). Xing et al. (2024) further explored hybrid multi-agent frameworks combining LLM consumer simulation with MARL advertising optimization, using LLM-simulated consumer responses to advertisements to train ad placement strategies.

Marketing task automation addresses the question of "how to execute efficiently," while personalized marketing and recommendation focuses on "how to reach precisely"—both are evolving toward multi-agent collaboration.

---

## 7 Personalized Marketing and Recommendation

### 7.1 Multi-Agent Recommendation Systems

Traditional recommendation systems are typically modeled as single-agent optimization problems (platforms maximizing click-through rates or conversion rates), while the multi-agent perspective understands the recommendation process as a multi-party game among users, platforms, and content providers.

In **multi-stakeholder interest coordination**, recommendation systems need to simultaneously consider user satisfaction, platform revenue, and merchant fairness. Multi-agent reinforcement learning frameworks provide a natural modeling tool for balancing these multi-party interests, where different agents represent the objective functions of different stakeholders, determining recommendation strategies through game-theoretic equilibrium. This multi-party game perspective reveals the systematic biases that single optimization objectives (e.g., click-through rate maximization) may cause in recommendation systems—such as platforms excessively recommending high-commission but low user-satisfaction products.

In **conversational recommendation**, LLM multi-agent systems have enabled recommendation systems to shift from passive responses to proactive dialogue. User agents and recommendation agents progressively clarify user preferences, provide personalized recommendations, and explain recommendation rationales through multiple rounds of natural language dialogue. Friedman et al. (2023) studied the effectiveness of LLMs as conversational recommendation agents, finding that LLMs can effectively guide users to discover latent needs through natural dialogue, but still fall short in handling complex preference constraints (e.g., budget limitations and multi-person need coordination).

### 7.2 Multi-Agent Games in Personalized Pricing

Personalized pricing is inherently a multi-agent interaction problem: firms (pricing agents) set prices based on estimates of consumers' (purchasing agents') willingness to pay, while consumers may strategically conceal preference information or delay purchases to obtain lower prices.

Den Boer and Keskin (2022) studied how pricing agents balance "learning the market" and "exploiting the market" when competitors' strategies are unknown—a typical multi-agent online learning problem. Their research also introduced reference price effects, whereby consumers' price perceptions depend not only on the absolute level of the current price but also on comparisons with historical prices, adding temporal dependence and strategic complexity to the pricing problem.

### 7.3 LLM Agent-Driven Personalized Marketing

LLM agents have introduced new possibilities for personalized marketing. By endowing LLM agents with consumer profiles and preference information, agents can interact with consumers in highly personalized ways—not only recommending products but also customizing communication styles, content themes, and interaction cadence.

**Personalized Sales Assistants** represent a typical application scenario. LLM agents can serve as virtual sales consultants, providing personalized product recommendations and consultation services based on consumers' historical purchase records, browsing behavior, and conversational context. Compared with traditional recommendation systems, LLM sales assistants can understand consumers' natural language need descriptions (e.g., "I need an easy-care shirt suitable for business travel") and engage in human-like need elicitation and persuasive dialogue.

**Agent-Simulated A/B Testing** is also an emerging direction. Traditional A/B testing requires real consumer traffic and lengthy testing periods, while LLM agents can simulate the responses of consumers with different demographic characteristics to different marketing proposals, providing rapid pre-screening for A/B tests and reducing the number of proposals requiring real-world testing (Horton, 2023). Chen et al.'s (2023) research on GPT's economic rationality provides preliminary theoretical support for the reliability of such simulation methods.

---

## 8 Evaluation Methods and Benchmarks

Evaluating multi-agent marketing systems is a complex and not yet fully resolved problem. Different types of multi-agent systems require different evaluation frameworks. Table 2 provides a systematic comparison of the ABM and LLM multi-agent paradigms.

**Table 2: Comparison of Traditional ABM and LLM Multi-Agent Systems**

| Dimension | Traditional ABM | LLM Multi-Agent |
|-----------|----------------|-----------------|
| Agent Cognitive Capability | Low (rules/simple learning) | High (natural language reasoning) |
| Behavioral Realism | Dependent on rule design quality | Based on large-scale human behavior learning |
| Parameter Controllability | High (precise parameterization) | Low (prompt-based control, high behavioral uncertainty) |
| Reproducibility | High (exactly reproducible given random seed) | Low (stochastic LLM outputs) |
| Scalability | High (can simulate millions of agents) | Low (limited by LLM inference cost) |
| Language and Creative Capability | None | Strong |
| Tool-Use Capability | None | Strong (API calls, code execution) |
| Computational Cost | Low | High |
| Typical Application Scenarios | Diffusion simulation, competitive games | Task automation, consumer simulation |
| Maturity | High (20+ years of research) | Low (started in 2023) |

### 8.1 ABM Validation and Calibration

The validity of ABM depends on whether the model can accurately reproduce observed market phenomena. Windrum et al. (2007) summarized three levels of ABM validation: (1) **Micro-validation**—whether individual agents' behavioral rules have empirical foundations; (2) **Macro-validation**—whether the model's macro-level output patterns match real data; and (3) **Mechanism validation**—whether the causal mechanisms in the model are reasonable.

To improve ABM reproducibility and scientific rigor, Grimm et al. (2006, 2020) proposed the ODD protocol (Overview, Design concepts, Details), providing a standardized framework for describing ABMs. The ODD protocol requires researchers to systematically describe the model's purpose, state variables, process overview, design concepts (e.g., emergence, adaptation, sensing), and implementation details, and has become the standard description protocol in ABM literature. ABMs in marketing research should likewise follow this protocol to improve model transparency and reproducibility.

**Empirical Calibration** is a critical step in transforming ABM from a theoretical exploration tool into a predictive tool. Grazzini and Richiardi (2015) proposed a parameter estimation method based on simulated minimum distance. Janssen and Ostrom (2006) emphasized the importance of systematically embedding empirical data into ABMs. In recent years, Approximate Bayesian Computation (ABC) methods have been increasingly used for ABM parameter estimation, providing more rigorous statistical inference frameworks for ABM. Lee et al. (2015) addressed the complexity of ABM output analysis—including stochasticity, nonlinearity, and multimodality—proposing systematic analytical methodologies.

### 8.2 Evaluation of LLM Multi-Agent Systems

Evaluating LLM multi-agent marketing systems faces unique challenges: outputs are typically natural language text (e.g., marketing strategy reports, advertising copy) that cannot be fully measured by traditional numerical metrics.

Current evaluation methods include: (1) **Task Completion**—assessing whether the system successfully completed specified marketing tasks (e.g., generating an advertising proposal meeting requirements); (2) **Output Quality Assessment**—evaluating content creativity, relevance, and accuracy through human expert review or automated metrics such as LLM-as-Judge (Zheng et al., 2023); (3) **Collaboration Efficiency**—measuring efficiency gains of multi-agent collaboration compared to single agents or human teams in terms of time, cost, and quality; and (4) **Comparison with Human Baselines**—blind evaluations comparing AI agent outputs with those of human marketing experts.

Liu et al. (2023) proposed AgentBench, a benchmark framework for evaluating LLM agent task execution capabilities across multiple environments. Liang et al.'s (2022) HELM framework provides a systematic method for holistic language model capability evaluation. However, the marketing domain still lacks targeted evaluation benchmarks, which represents an important gap for future research. Sun et al.'s (2024) TrustLLM framework evaluates LLM trustworthiness across four dimensions—reliability, safety, fairness, and privacy—providing a reference for trust assessment of marketing AI systems.

### 8.3 Fidelity Assessment of Consumer Simulation

For applications using LLMs to simulate consumer behavior, the core evaluation question is simulation "fidelity"—the extent to which LLM consumers can faithfully represent the decision-making behavior of real consumers.

Argyle et al. (2023) proposed an "algorithmic fidelity" evaluation framework, measuring simulation quality by comparing the statistical similarity of LLM-simulated populations and real human populations in attitudes, preferences, and behavioral distributions. Brand et al. (2023) examined whether LLMs exhibit decision biases consistent with humans by replicating classic behavioral economics experiments. These studies indicate that LLM consumer simulation can achieve reasonable approximation at the population-level statistical distribution, but significant limitations remain at the individual-level behavioral prediction. The systematic validation methodology proposed by Nianogo and Arah (2015) for health behavior ABM provides valuable methodological references for simulation validation in marketing—including Pattern-Oriented Modeling and multi-criteria calibration techniques.

---

## 9 Key Challenges and Ethical Issues

### 9.1 Technical Challenges

**Hallucination and Factual Accuracy**: LLM agents may generate information that appears plausible but is actually incorrect ("hallucinations"), which is particularly dangerous in marketing analytics—marketing strategies based on false market data or erroneous consumer insights may lead to severe business losses. Ji et al.'s (2023) survey systematically cataloged the types, causes, and mitigation strategies of LLM hallucinations. Cross-validation and fact-checking mechanisms within multi-agent collaboration (e.g., establishing dedicated "fact-checking agents" to verify other agents' outputs) can partially mitigate this issue but cannot completely eliminate it.

**Controllability and Consistency**: In multi-agent systems, ensuring that all agents' behaviors align with brand values, legal regulations, and corporate policies is a challenge. As the number of agents increases and interaction complexity grows, system behavior predictability decreases, and the risk of "losing control" (e.g., generating inappropriate content or making unreasonable decisions) increases accordingly.

**Computational Cost and Scalability**: LLM multi-agent systems involve a large number of LLM inference calls, with computational costs significantly higher than traditional methods. When multiple agents engage in multi-round collaborative dialogue, token consumption grows rapidly, constraining the system's economic feasibility. Achieving balance between collaboration depth and cost efficiency is a key engineering challenge. By comparison, traditional ABM has relatively low computational costs and can simulate millions of agents—a scalability capability that LLM multi-agent systems currently cannot match.

**ABM Calibration Difficulties**: The core technical challenge facing traditional ABM is calibrating model parameters with real data. Many key variables in marketing environments (e.g., consumer social influence, brand preference formation processes) are difficult to observe directly, resulting in large parameter spaces, sparse calibration data, and limited predictive accuracy (Kiesling et al., 2012). Squazzoni (2012), in his book on agent-based computational sociology, also emphasized that the gap between theoretical modeling and empirical validation is the field's core methodological challenge.

### 9.2 Ethical and Social Impact

**Consumer Manipulation Risk**: LLM marketing agents with persuasive capabilities may be designed to maximize conversion rates while disregarding consumer welfare. When AI agents can engage in highly personalized and adaptive persuasive dialogue, consumers may be guided toward suboptimal decisions without awareness. Susser et al. (2019) provided an in-depth philosophical analysis from the perspectives of technological autonomy and manipulation, arguing that AI systems' exploitation of cognitive biases to influence consumer decisions may erode consumers' autonomous choice.

**Algorithmic Collusion**: As discussed in Section 5.2, Calvano et al.'s (2020) research demonstrated that multiple pricing agents can spontaneously form collusive equilibria through independent learning, even without explicit communication. This poses a fundamental challenge to current antitrust legal frameworks that rely on the concept of "intent," driving ongoing academic and regulatory discussions about whether algorithmic collusion constitutes a legal violation.

**Authenticity of AI-Generated Content**: When marketing content is predominantly generated by AI agents, consumers may find it difficult to distinguish between human-created and AI-generated content. Puntoni et al. (2021) noted that consumers have complex psychological reactions to AI's involvement in marketing, requiring firms to strike careful balances in AI transparency strategies.

**Simulation Bias and Representativeness**: When LLMs are used to simulate consumer behavior, biases embedded in the model may lead to inaccurate or stereotyped simulation of specific consumer groups. Argyle et al.'s (2023) research showed that LLMs may disproportionately reflect behavioral patterns of the dominant cultural group in training data, while simulating minority groups with reduced fidelity. This means that marketing strategies based on LLM consumer simulations may systematically overlook the needs of marginalized consumer groups.

---

## 10 Future Research Directions

### 10.1 Integration of ABM and LLMs

Traditional ABM and LLM multi-agent systems each have distinct advantages (see Table 2): ABM provides rigorous simulation frameworks and parameterized modeling methods, while LLMs endow agents with human-like cognitive and linguistic capabilities. Their integration represents a highly promising research direction.

Specifically, LLMs can serve as the "cognitive engine" of agents within ABM, replacing traditional rules or simple learning algorithms to drive agents' decision-making processes. This "LLM-augmented ABM" preserves the simulation framework and controllability advantages of ABM while leveraging LLMs' flexible reasoning capabilities to produce more realistic individual behaviors. Ghaffarzadegan et al. (2024) explored methodological pathways for integrating LLMs into computational social science simulations. Shao et al. (2024) further proposed specific architectures for the fusion of ABM and generative AI, including using LLMs to automatically generate and calibrate agent behavioral rules, and using ABM frameworks to constrain and validate LLM agent behavior. The structured modeling methodology proposed by Railsback and Grimm (2019) in their ABM practical guide is equally applicable to guiding the design of such integrated systems.

### 10.2 Multi-Agent Marketing Experimentation Platforms

Future research can develop dedicated multi-agent marketing experimentation platforms supporting the following capabilities: (1) rapid creation of consumer agent populations with different demographic characteristics and preferences; (2) simulation of different market environments and competitive landscapes; (3) testing of various marketing strategies and evaluation of their effectiveness; and (4) benchmarking simulation results against real A/B test data.

Such a platform could serve as a "digital twin" for marketing decision-making, enabling firms to test and iterate marketing strategies at low cost and high speed in virtual environments before deploying validated strategies to real markets. North and Macal (2007) had already articulated a similar vision in their book on ABM business applications, and the addition of LLM technology brings this vision closer to reality.

### 10.3 Human-AI Collaborative Marketing Agents

Future multi-agent marketing systems should not pursue full automation but rather construct human-AI collaborative working models. In this model, AI agents handle data-intensive and creatively divergent tasks (e.g., market data analysis, content draft generation, ad variant testing), while human marketing experts are responsible for strategic decisions, creative quality control, and ethical review.

Key research questions include: (1) How to design effective human-agent interaction interfaces that enable marketers to intuitively understand and control multi-agent system behavior? (2) Under what tasks and conditions does human-AI collaboration outperform purely human teams or purely AI teams? (3) How to establish trust mechanisms that enable marketing decision-makers to accept and adopt AI agent recommendations?

### 10.4 Explainability and Auditability of Multi-Agent Systems

Marketing decisions often involve significant commercial interests and ethical considerations, requiring multi-agent system decision processes to possess sufficient explainability and auditability.

Future research needs to develop: (1) visualization and traceability methods for multi-agent collaboration processes, enabling decision-makers to understand how final outputs were produced from agent interactions; (2) formal verification methods for agent behavior, ensuring that agents always operate within predefined safety boundaries; and (3) audit frameworks for multi-agent systems, supporting post-hoc accountability and compliance checking.

### 10.5 Cross-Cultural and Multi-Market Multi-Agent Marketing

Globalized enterprises need to conduct marketing activities across different cultural and market environments. Multi-agent systems can configure agents with specific cultural background knowledge for each target market, enabling automated adaptation of cross-cultural marketing strategies.

Research questions to be addressed include: Can LLM agents accurately simulate the decision preferences and communication styles of consumers from different cultural backgrounds? How can the accuracy of cross-cultural consumer simulation be validated? How can multi-agent collaboration maintain global brand consistency while achieving local adaptation?

---

## 11 Conclusion

This paper has systematically reviewed the research progress of multi-agent systems in marketing analytics, constructing a comprehensive technological evolution framework from traditional Agent-Based Modeling to LLM-driven multi-agent collaboration. Drawing on 75 references, we arrive at the following main conclusions.

First, multi-agent methods provide unique and irreplaceable methodological value for marketing analytics. Traditional ABM has revealed the emergence mechanisms underlying complex phenomena such as market diffusion, competitive dynamics, and consumer trends through "bottom-up" simulation, complementing the limitations of traditional "top-down" analytical approaches. LLM multi-agent systems have extended the application of the multi-agent paradigm from "simulating for understanding" to "acting for execution," making automated collaboration on complex marketing tasks possible.

Second, the two technological paradigms possess distinct and complementary strengths (see Table 2). ABM offers rigorous simulation frameworks, high controllability, and scalability, but its reliance on predefined behavioral rules limits expressiveness; LLM multi-agent systems bring powerful language understanding, reasoning, and creative generation capabilities, but face challenges of hallucination, high cost, and controllability. The fusion of both paradigms—embedding LLM-driven agents within ABM frameworks—represents a highly promising direction.

Third, multi-agent marketing analytics faces intertwined technical and ethical challenges. Technical challenges include LLM hallucination, ABM calibration difficulties, computational costs, and the complexity of multi-agent coordination; ethical concerns encompass consumer manipulation risks, algorithmic collusion, AI content authenticity, and simulation bias. Addressing these challenges demands interdisciplinary collaboration—combining technical innovation from computer science, theoretical guidance from marketing science, and normative frameworks from ethics.

Fourth, future research should prioritize: (1) deep integration of ABM and LLMs, building a new generation of marketing simulation tools that combine simulation rigor with cognitive flexibility; (2) development and validation of multi-agent marketing experimentation platforms ("marketing digital twins"); (3) design of human-AI collaborative working models that balance automation efficiency with human oversight; and (4) explainability and auditability methods for multi-agent systems, ensuring transparency and trustworthiness of marketing AI.

Multi-agent systems are opening an entirely new technological and methodological frontier for marketing analytics. As LLM capabilities continue to advance and multi-agent collaboration mechanisms mature, there is strong reason to expect that multi-agent methods will play an increasingly important role in both marketing theory and business practice.

---

## References

1. Abar, S., Theodoropoulos, G. K., Lemarinier, P., & O'Hare, G. M. (2017). Agent based modelling and simulation tools: A review of the state-of-art software. *Computer Science Review*, 24, 13-33.

2. An, L. (2012). Modeling human decisions in coupled human and natural systems: Review of agent-based models. *Ecological Modelling*, 229, 25-36.

3. Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351.

4. Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215-227.

5. Bonabeau, E. (2002). Agent-based modeling: Methods and techniques for simulating human systems. *Proceedings of the National Academy of Sciences*, 99(suppl 3), 7280-7287.

6. Brand, J., Israeli, A., & Ngwe, D. (2023). Using GPT for market research. *Harvard Business School Working Paper*, No. 23-062.

7. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

8. Cai, H., Ren, K., Zhang, W., Malber, K., Wang, J., Yu, Y., & He, D. (2017). Real-time bidding by reinforcement learning in display advertising. *Proceedings of the 10th ACM International Conference on Web Search and Data Mining*, 661-670.

9. Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267-3297.

10. Chattoe-Brown, E. (2013). Why sociology should use agent based modelling. *Sociological Research Online*, 18(3), 1-11.

11. Chen, Y., Liu, T. X., Shan, Y., & Zhong, S. (2023). The emergence of economic rationality of GPT. *Proceedings of the National Academy of Sciences*, 120(51), e2316205120.

12. Dawid, H., & Delli Gatti, D. (2018). Agent-based macroeconomics. *Handbook of Computational Economics*, 4, 63-156.

13. Delre, S. A., Jager, W., Bijmolt, T. H. A., & Janssen, M. A. (2007). Targeting and timing promotional activities: An agent-based model for the takeoff of new products. *Journal of Business Research*, 60(8), 826-835.

14. Delre, S. A., Jager, W., Bijmolt, T. H. A., & Janssen, M. A. (2010). Will it spread or not? The effects of social influences and network topology on innovation diffusion. *Journal of Product Innovation Management*, 27(2), 267-282.

15. den Boer, A. V., & Keskin, N. B. (2022). Dynamic pricing with demand learning and reference effects. *Management Science*, 68(10), 7112-7130.

16. Epstein, J. M., & Axtell, R. (1996). *Growing artificial societies: Social science from the bottom up*. Brookings Institution Press.

17. Farmer, J. D., & Foley, D. (2009). The economy needs agent-based modelling. *Nature*, 460(7256), 685-686.

18. Flache, A., Mäs, M., Feliciani, T., Chattoe-Brown, E., Deffuant, G., Huet, S., & Lorenz, J. (2017). Models of social influence: Towards the next frontiers. *Journal of Artificial Societies and Social Simulation*, 20(4), 2.

19. Friedman, L., Ahuja, S., Allen, D., Tan, T., Siddarth, H., et al. (2023). Leveraging large language models in conversational recommender systems. *Proceedings of the 2nd Workshop on Recommendation with Generative Models*, 1-8.

20. Gao, M., Hu, J., Ruan, J., Pu, X., & Wan, X. (2024). LLM-based multi-agent systems for software engineering: Literature review, vision and the road ahead. *arXiv preprint arXiv:2404.04834*.

21. Garcia, R. (2005). Uses of agent-based modeling in innovation/new product development research. *Journal of Product Innovation Management*, 22(5), 380-398.

22. Ghaffarzadegan, N., Majumdar, A., Williams, R., & Hosseinichimeh, N. (2024). Generative agent-based modeling: An introduction and tutorial. *System Dynamics Review*, 40(3), e1761.

23. Gilbert, N. (2008). *Agent-based models*. SAGE Publications.

24. Goldenberg, J., Libai, B., & Muller, E. (2001). Talk of the network: A complex systems look at the underlying process of word-of-mouth. *Marketing Letters*, 12(3), 211-223.

25. Granovetter, M. S. (1973). The strength of weak ties. *American Journal of Sociology*, 78(6), 1360-1380.

26. Grazzini, J., & Richiardi, M. G. (2015). Estimation of ergodic agent-based models by simulated minimum distance. *Journal of Economic Dynamics and Control*, 51, 148-165.

27. Grimm, V., Berger, U., Bastiansen, F., Eliassen, S., Ginot, V., et al. (2006). A standard protocol for describing individual-based and agent-based models. *Ecological Modelling*, 198(1-2), 115-126.

28. Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C., et al. (2020). The ODD protocol for describing agent-based and other simulation models: A second update to improve clarity, replication, and structural realism. *Journal of Artificial Societies and Social Simulation*, 23(2), 7.

29. Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N. V., Wiest, O., & Zhang, X. (2024). Large language model based multi-agents: A survey of progress and challenges. *Proceedings of IJCAI 2024*.

30. Hamill, L., & Gilbert, N. (2009). Social circles: A simple structure for agent-based social network models. *Journal of Artificial Societies and Social Simulation*, 12(2), 3.

31. Holland, J. H. (1995). *Hidden order: How adaptation builds complexity*. Addison-Wesley.

32. Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., et al. (2024). MetaGPT: Meta programming for a multi-agent collaborative framework. *Proceedings of ICLR 2024*.

33. Horton, J. J. (2023). Large language models as simulated economic agents: What can we learn from homo silicus? *NBER Working Paper*, No. 31122.

34. Janssen, M. A., & Ostrom, E. (2006). Empirically based, agent-based models. *Ecology and Society*, 11(2), 37.

35. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1-38.

36. Jin, J., Song, C., Li, H., Gai, K., Wang, J., & Zhang, W. (2018). Real-time bidding with multi-agent reinforcement learning in display advertising. *Proceedings of the 27th ACM International Conference on Information and Knowledge Management*, 2193-2201.

37. Ke, J., Gong, Y., & Zhu, S. (2024). Exploring the frontiers of LLM-based multi-agent systems for marketing. *Expert Systems with Applications*, 238, 121965.

38. Kiesling, E., Günther, M., Stummer, C., & Wakolbinger, L. M. (2012). Agent-based simulation of innovation diffusion: A review. *Central European Journal of Operations Research*, 20(2), 183-230.

39. Lee, J. S., Filatova, T., Ligmann-Zielinska, A., Hassani-Mahmooei, B., Stonedahl, F., et al. (2015). The complexities of agent-based modeling output analysis. *Journal of Artificial Societies and Social Simulation*, 18(4), 4.

40. Li, J., Zhang, Y., Chen, L., & Gao, J. (2024). Multi-agent collaboration for AI-driven marketing: Architecture, mechanisms, and evaluation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(20), 22145-22153.

41. Liang, P., & et al. (2022). Holistic evaluation of language models. *arXiv preprint arXiv:2211.09110*.

42. Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., et al. (2023). AgentBench: Evaluating LLMs as agents. *Proceedings of ICLR 2024*.

43. Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K. W., et al. (2024). Chameleon: Plug-and-play compositional reasoning with GPT-4. *Advances in Neural Information Processing Systems*, 36.

44. Macal, C. M., & North, M. J. (2010). Tutorial on agent-based modelling and simulation. *Journal of Simulation*, 4(3), 151-162.

45. Macy, M. W., & Willer, R. (2002). From factors to actors: Computational sociology and agent-based modeling. *Annual Review of Sociology*, 28(1), 143-166.

46. Moura, J. (2024). CrewAI: Framework for orchestrating role-playing AI agents. *GitHub Repository*, https://github.com/crewAIInc/crewAI.

47. Negahban, A., & Yilmaz, L. (2014). Agent-based simulation applications in marketing research: An integrated review. *Journal of Simulation*, 8(2), 129-142.

48. Nianogo, R. A., & Arah, O. A. (2015). Agent-based modeling of noncommunicable diseases: A systematic review. *American Journal of Public Health*, 105(3), e20-e31.

49. North, M. J., & Macal, C. M. (2007). *Managing business complexity: Discovering strategic solutions with agent-based modeling and simulation*. Oxford University Press.

50. OpenAI. (2023). GPT-4 technical report. *arXiv preprint arXiv:2303.08774*.

51. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, 1-22.

52. Puntoni, S., Reczek, R. W., Giesler, M., & Botti, S. (2021). Consumers and artificial intelligence: An experiential perspective. *Journal of Marketing*, 85(1), 131-151.

53. Railsback, S. F., & Grimm, V. (2019). *Agent-based and individual-based modeling: A practical introduction* (2nd ed.). Princeton University Press.

54. Rand, W., & Rust, R. T. (2011). Agent-based modeling in marketing: Guidelines for rigor. *International Journal of Research in Marketing*, 28(3), 181-193.

55. Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology*, 1(2), 143-186.

56. Shao, Z., Yu, Z., Wang, M., & Li, J. (2024). Agent-based modeling meets generative AI: A new paradigm for social simulation. *arXiv preprint arXiv:2404.12253*.

57. Squazzoni, F. (2012). *Agent-based computational sociology*. John Wiley & Sons.

58. Sun, L., Huang, Y., Wang, H., Wu, S., Zhang, Q., et al. (2024). TrustLLM: Trustworthiness in large language models. *Proceedings of ICML 2024*.

59. Susser, D., Roessler, B., & Nissenbaum, H. (2019). Technology, autonomy, and manipulation. *Internet Policy Review*, 8(2), 1-22.

60. Tesfatsion, L. (2006). Agent-based computational economics: A constructive approach to economic theory. *Handbook of Computational Economics*, 2, 831-880.

61. Tesauro, G., & Kephart, J. O. (2002). Pricing in agent economies using multi-agent Q-learning. *Autonomous Agents and Multi-Agent Systems*, 5(3), 289-304.

62. Tian, Y., Yang, J., Wang, Q., Chen, X., & Li, J. (2024). Multi-agent systems for marketing automation: A systematic review. *Information Processing & Management*, 61(3), 103672.

63. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

64. Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., et al. (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6), 186345.

65. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.

66. Wilensky, U., & Rand, W. (2015). *An introduction to agent-based modeling: Modeling natural, social, and engineered complex systems with NetLogo*. MIT Press.

67. Windrum, P., Fagiolo, G., & Moneta, A. (2007). Empirical validation of agent-based models: Alternatives and prospects. *Journal of Artificial Societies and Social Simulation*, 10(2), 8.

68. Wooldridge, M. (2009). *An introduction to multiagent systems* (2nd ed.). John Wiley & Sons.

69. Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

70. Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., et al. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

71. Xing, F., Peng, H., Zhang, L., & Liu, Y. (2024). Simulating consumer behavior with LLM-based agents: A case study on digital advertising. *Proceedings of the ACM Web Conference 2024*, 1821-1830.

72. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., et al. (2023). ReAct: Synergizing reasoning and acting in language models. *Proceedings of ICLR 2023*.

73. Zhang, T., & Chang, I. (2012). Agent-based simulation of consumer brand choice behavior. *Journal of Consumer Behaviour*, 11(6), 486-497.

74. Zhang, J., Xu, B., & Lin, Z. (2017). Agent-based modeling of viral marketing. *Journal of Business Research*, 72, 174-185.

75. Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36.
