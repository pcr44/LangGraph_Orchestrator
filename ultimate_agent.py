import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import operator
import time
import xml.etree.ElementTree as ET
import markdown
from io import BytesIO
from docx import Document
from htmldocx import HtmlToDocx
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, Field

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PAGE CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(page_title="Agentic Oncology Orchestrator", layout="wide")
st.title("🧬 Agentic Precision Oncology Pipeline")
st.markdown("Powered by LangGraph, PyDESeq2, OncoKB, and PubMed")

try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    oncokb_key = st.secrets["ONCOKB_API_KEY"]
except KeyError:
    st.error("⚠️ Secrets not found! Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()

# --- INITIALIZE SESSION STATE (MEMORY) ---
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. GRAPH STATE & SCHEMAS
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    significant_genes: List[Dict[str, Any]]
    plan: List[str]
    gathered_evidence: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Step-by-step plan of tools to execute.")

# ==========================================
# 2. THE TOOLS (Python Functions)
# ==========================================
def get_onco_data(hugo, alteration, tumor_type):
    url = "https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange"
    params = {"hugoSymbol": hugo, "alteration": alteration, "tumorType": tumor_type}
    headers = {"accept": "application/json", "Authorization": f"Bearer {oncokb_key}"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            treatments = data.get('treatments', [])
            if not treatments: return {"status": "No drug entries found."}
            
            results = []
            for treatment in treatments:
                drugs = [d.get('drugName', '') for d in treatment.get('drugs', [])]
                results.append({
                    "drugName": ", ".join(drugs), 
                    "levelOfEvidence": treatment.get('level', 'Unknown'),
                    "pmids": treatment.get('pmids', [])
                })
            return {"status": "Success", "drugs": results}
        return {"status": f"OncoKB Error: {response.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def search_pubmed(gene, tumor_type):
    search_query = f"{gene}[Gene] AND {tumor_type} AND targeted therapy"
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": search_query, "retmode": "json", "retmax": 3}
    
    try:
        res = requests.get(search_url, params=search_params)
        if res.status_code != 200:
            return {"status": f"PubMed Search Error: {res.status_code}"}
            
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return {"status": "No experimental literature found."}
            
        time.sleep(0.5) 
        
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        
        fetch_res = requests.get(fetch_url, params=fetch_params)
        if fetch_res.status_code != 200:
            return {"status": f"PubMed Fetch Error: {fetch_res.status_code}"}
            
        papers = []
        root = ET.fromstring(fetch_res.content)
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else "Unknown"
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else "No Title"
            
            abstract_text = ""
            abstract_nodes = article.findall('.//AbstractText')
            if abstract_nodes:
                abstract_text = " ".join([node.text for node in abstract_nodes if node.text])
            else:
                abstract_text = "No abstract available."
                
            papers.append({
                "PMID": pmid, 
                "Title": title,
                "Abstract": abstract_text[:1000]
            })
            
        time.sleep(0.5)
        return {"status": "Success", "papers": papers}
        
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

def search_clinical_trials(gene, tumor_type):
    url = "https://clinicaltrials.gov/api/v2/studies"
    query = f"{gene} AND {tumor_type}"
    params = {"query.cond": query, "filter.overallStatus": "RECRUITING", "pageSize": 3}
    
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json()
            studies = data.get("studies", [])
            if not studies:
                return {"status": "No recruiting trials found."}
                
            trials = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                design = protocol.get("designModule", {}) 
                
                nct_id = ident.get("nctId", "Unknown NCT")
                title = ident.get("briefTitle", "No Title")
                phase = ", ".join(design.get("phases", ["Phase Unknown"])) 
                
                trials.append({"NCT_ID": nct_id, "Title": title, "Phase": phase})
                
            time.sleep(0.5)
            return {"status": "Success", "trials": trials}
            
        return {"status": f"ClinicalTrials Error: {res.status_code}"}
    except Exception as e:
        return {"status": f"Request failed: {str(e)}"}

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def planner_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(Plan)
    
    sys_msg = """You are an expert Clinical Bioinformatics Planner. 
    Analyze the user prompt and genes. Output a step-by-step plan to gather data.
    Available Tools: 
    1. 'OncoKB' (FDA drugs)
    2. 'PubMed' (Experimental research)
    3. 'ClinicalTrials' (Actively recruiting trials)"""
    
    context = f"User Prompt: {state.get('user_prompt')}\nGenes: {state.get('significant_genes')}"
    response = structured_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=context)])
    
    return {"plan": response.steps}

def executor_node(state: AgentState):
    plan_text = " ".join(state.get("plan", [])).lower()
    genes = state.get("significant_genes", [])
    new_evidence = []
    
    for gene_info in genes:
        hugo = gene_info.get("hugo")
        alt = gene_info.get("alteration")
        tumor_type = gene_info.get("tumor_type")
        
        # --- NEW: We extract the source tag we created in Section 4 ---
        source_tag = gene_info.get("source", "Unknown Source")
        
        # --- NEW: We pass the source tag directly into the AI's evidence clipboard ---
        report = {"gene": hugo, "alteration": alt, "source": source_tag, "evidence": {}}
        
        if "oncokb" in plan_text:
            report["evidence"]["OncoKB"] = get_onco_data(hugo, alt, tumor_type)
        if "pubmed" in plan_text:
            report["evidence"]["PubMed"] = search_pubmed(hugo, tumor_type)
        if "clinicaltrials" in plan_text or "trials" in plan_text:
            print(f"   -> Fetching Clinical Trials for {hugo}...")
            report["evidence"]["ClinicalTrials"] = search_clinical_trials(hugo, tumor_type)
            
        new_evidence.append(report)
        
    return {"gathered_evidence": new_evidence}

def writer_node(state: AgentState):
    print("✍️ [NODE: Writer] Synthesizing the final clinical report...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_key)
    
    sys_msg = """You are an expert Clinical Oncology Medical Writer.
    Write a beautiful, multi-paragraph scientific report answering the user's prompt.
    
    CRITICAL CLINICAL TRIAGE RULES:
    1. Standard of Care (On-Label): List only drugs with Level 1 or Level 2 evidence. These are FDA-approved for the user's specific cancer type.
    2. Repurposing Opportunities (Off-Label): List only drugs with Level 3 (3A, 3B) or Level 4 evidence. These are approved for different cancers but show biomarker matches.
    
    CRITICAL GUARDRAILS: 
    1. ONLY use the data in the Gathered Evidence. Do NOT invent drugs or trials.
    2. For PubMed literature, read the Abstracts. Write a 1-2 sentence clinical summary of what the study actually found. Cite the PMID.
    3. Make all ClinicalTrials NCT IDs clickable markdown links.
    4. Important Context: Because this data originates from RNA sequencing (Overexpression), add a brief, single-sentence disclaimer at the top of the report stating: "*Note: Targeted therapies listed below typically require DNA confirmation of the specific mutation (e.g., V600E, L858R) associated with the overexpressed gene.*"
    
    REQUIRED REPORT STRUCTURE:
    You MUST format your report using this EXACT Markdown structure for EVERY gene sequentially. Do not omit the PMIDs:
    
    ## [Gene Name] ([Alteration])
    
    ### 💊 OncoKB Therapeutics
    - **Standard of Care (On-Label):**
      - [Drug Name]: [Clinical Context] (PMIDs: [List PMIDs])
    - **Repurposing Opportunities (Off-Label):**
      - [Drug Name]: [Clinical Context] (PMIDs: [List PMIDs])
    
    ### 🔬 Experimental Literature
    - **[Study Topic/Focus]:** [1-2 sentence summary of abstract] (PMID: [Number])
    
    ### 🏥 Actively Recruiting Trials
    - **[[NCT ID]](https://clinicaltrials.gov/study/[NCT ID]):** [Phase] - [Trial Title]
    
    Do not deviate from this structure."""
    
    user_context = f"User Prompt: {state.get('user_prompt')}\nGathered Evidence: {json.dumps(state.get('gathered_evidence'))}"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_context)
    ])
    
    print("✅ Final report successfully written.")
    return {"final_report": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "writer")
workflow.add_edge("writer", END)

orchestrator = workflow.compile()

# ==========================================
# 4. STREAMLIT UI & MAIN EXECUTION
# ==========================================
st.subheader("1. Configure Analysis")
col1, col2 = st.columns(2) # Back to a clean 2-column layout!

with col1:
    st.markdown("**Transcriptomics (RNA-Seq)**")
    counts_file = st.file_uploader("Upload RNA Counts (CSV)", type=["csv"])
    metadata_file = st.file_uploader("Upload Metadata (CSV)", type=["csv"])

with col2:
    st.markdown("**Clinical Context**")
    disease_interest = st.text_input("Cancer Type", value="Melanoma")
    user_prompt = st.text_area(
        "📝 AI Instructions", 
        value="Identify potential therapeutics for the top targets. Highlight experimental PubMed literature."
    )
    
    # Your requested optional dropdown menu with useful external links!
    with st.expander("🧬 External DNA & Mutation Resources (Optional)"):
        st.write("Cross-reference your RNA findings with known DNA mutational databases:")
        st.markdown("- [cBioPortal for Cancer Genomics](https://www.cbioportal.org/)")
        st.markdown("- [COSMIC (Catalogue of Somatic Mutations in Cancer)](https://cancer.sanger.ac.uk/cosmic)")
        st.markdown("- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)")

if st.button("Run RNA-to-Clinical Pipeline", type="primary"):
    if counts_file and metadata_file:
        agent_payload = []
        
        counts_df = pd.read_csv(counts_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        with st.status("⚙️ Processing RNA-Seq Data...", expanded=True) as status:
            st.write("Running PyDESeq2 Differential Expression...")
            dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors="condition")
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
            stat_res.summary()
            results_df = stat_res.results_df
            status.update(label="PyDESeq2 Complete!", state="complete", expanded=False)
            
        # Volcano Plot
        plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
        plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
        plot_df['Log2FC'] = plot_df['log2FoldChange'].round(2)
        plot_df['P-value (adj)'] = plot_df['padj'].apply(lambda x: f"{x:.2e}")
        
        conditions = [(plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] > 2), (plot_df['padj'] < 0.05) & (plot_df['log2FoldChange'] < -2)]
        choices = ['Upregulated (Target)', 'Downregulated']
        plot_df['Significance'] = np.select(conditions, choices, default='Not Significant')
        
        fig = px.scatter(
            plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', 
            color_discrete_map={'Upregulated (Target)': 'red', 'Downregulated': 'blue', 'Not Significant': 'lightgrey'}, 
            hover_name=plot_df.index, hover_data={'log2FoldChange': False, '-log10(padj)': False, 'Log2FC': True, 'P-value (adj)': True}
        )
        st.session_state.volcano_fig = fig

        # THE SMART LOOKUP DICTIONARY
        # If the RNA is highly active, we assume the most common actionable mutation to get rich OncoKB data
        mutation_lookup = {
            "BRAF": "V600E",
            "EGFR": "L858R",
            "KRAS": "G12C",
            "PIK3CA": "H1047R",
            "ERBB2": "Amplification"
        }

        # Extract top 3 RNA targets
        sig_genes = results_df[(results_df['padj'] < 0.05) & (results_df['log2FoldChange'] > 2)].sort_values(by='padj').head(3)
        for g in sig_genes.index:
            gene_str = str(g).strip().upper()
            
            # Check our smart dictionary. If not found, default to Amplification.
            assumed_alt = mutation_lookup.get(gene_str, "Amplification")
            
            agent_payload.append({
                "hugo": gene_str, 
                "alteration": assumed_alt, 
                "tumor_type": disease_interest,
                "source": "RNA Overexpression"
            })

        # --- LANGGRAPH EXECUTION ---
        st.markdown("---")
        st.subheader("🧠 AI Orchestration")
        with st.spinner(f"The AI is building a clinical profile for {len(agent_payload)} RNA targets..."):
            initial_state = {
                "user_prompt": user_prompt,
                "significant_genes": agent_payload,
                "plan": [],
                "gathered_evidence": [],
                "final_report": ""
            }
            final_state = orchestrator.invoke(initial_state)
            
            st.session_state.plan = final_state["plan"]
            st.session_state.final_report = final_state["final_report"]
            st.session_state.run_complete = True
            st.session_state.messages = [] 
            
    else:
        st.warning("Please upload RNA-Seq files (Counts + Metadata) to begin.")

# ==========================================
# 5. RENDER RESULTS & CHATBOT (From Memory)
# ==========================================
if st.session_state.run_complete:
    st.markdown("---")
    st.subheader("📈 Gene Expression Volcano Plot")
    st.plotly_chart(st.session_state.volcano_fig, use_container_width=True)
    
    with st.expander("🔍 View the AI's Strategic Plan"):
        for step in st.session_state.plan:
            st.write(f"- {step}")
            
    st.markdown("### 📄 Final Synthesized Clinical Report")
    st.info("This report was autonomously written by the Medical Writer LLM based solely on validated tool data.")
    st.markdown(st.session_state.final_report)
    
    # --- EXPORT MENU (HTML & DOCX) ---
    st.markdown("### 💾 Export Options")
    
    # 1. Prepare the Base HTML
    html_content = markdown.markdown(st.session_state.final_report, extensions=['tables'])
    
    # 2. Build the Styled HTML File
    styled_html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 40px auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                ul {{ margin-bottom: 20px; }}
                li {{ margin-bottom: 8px; }}
            </style>
        </head>
        <body>
            <h1>Clinical AI Orchestrator Report</h1>
            <p><strong>Disease Target:</strong> {disease_interest}</p>
            <hr>
            {html_content}
        </body>
    </html>
    """
    
    # 3. Build the Word Document (.docx) in Memory
    doc = Document()
    doc.add_heading(f'Clinical AI Orchestrator Report - {disease_interest}', level=1)
    
    # Use HtmlToDocx to cleanly convert our markdown-generated HTML into Word format
    parser = HtmlToDocx()
    parser.add_html_to_document(html_content, doc)
    
    # Save the Word doc to a virtual memory buffer
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0) # Reset the buffer pointer so Streamlit can read it from the beginning
    
    # 4. Display the Buttons Side-by-Side
    col_down1, col_down2 = st.columns(2)
    
    with col_down1:
        st.download_button(
            label="🌐 Download as HTML (Browser/PDF)",
            data=styled_html,
            file_name=f"{disease_interest}_Clinical_Report.html",
            mime="text/html",
            use_container_width=True
        )
        
    with col_down2:
        st.download_button(
            label="📄 Download as Word Document (.docx)",
            data=doc_buffer,
            file_name=f"{disease_interest}_Clinical_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )

    # --- INTERACTIVE CHATBOT ---
    st.markdown("---")
    st.subheader("💬 Discuss the Findings")
    st.write("Ask follow-up questions about the clinical trials, specific drugs, or resistance mechanisms mentioned above.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input
    if prompt := st.chat_input("E.g., What is the mechanism of action for CL-387785?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_key)
                
                # Give the chat LLM the final report as its core context
                chat_sys_msg = f"You are a helpful oncology assistant. Answer the user's questions strictly based on the following report:\n\n{st.session_state.final_report}"
                
                messages = [SystemMessage(content=chat_sys_msg)]
                for m in st.session_state.messages:
                    if m["role"] == "user": messages.append(HumanMessage(content=m["content"]))
                    else: messages.append(AIMessage(content=m["content"]))
                    
                response = chat_llm.invoke(messages)
                st.markdown(response.content)
                
        st.session_state.messages.append({"role": "assistant", "content": response.content})