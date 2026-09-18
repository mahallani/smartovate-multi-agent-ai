# 🤖 Smartovate Multi-Agent AI System

A multi-agent Generative AI system developed during an internship at **Smartovate**.  
The project combines specialized AI agents, Machine Learning, Azure OpenAI and Docker to automate a workflow from task planning to code generation, review and execution.

---

## 🎯 Project Overview

The objective of this project is to design and develop a collaborative AI system where several specialized agents work together to transform a user's task into a validated and executable solution.

The system combines:

- 🧠 **Multi-Agent AI**
- ☁️ **Azure OpenAI**
- 🤖 **AutoGen**
- 📊 **Machine Learning**
- 🐳 **Docker**
- 🖥️ **Streamlit**
- 🐍 **Python**

---

## 🏗️ System Architecture

The main workflow is:

```text
                    ┌──────────────┐
                    │     User     │
                    │     Task     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Planner Agent  │
                  │   Task Planning │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Coder Agent   │
                  │ Code Generation  │
                  └────────┬────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   ML Risk Predictor  │
                │ Revision Risk        │
                │ Prediction            │
                └──────────┬───────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │ Reviewer Agent  │
                 │ Validation      │
                 └───────┬─────────┘
                         │
                 ┌───────┴────────┐
                 │                │
              Approved         Revision
                 │                │
                 │                └──────► Coder
                 │
                 ▼
          ┌──────────────────┐
          │ Docker Executor  │
          │ Isolated Runtime │
          └────────┬─────────┘
                   │
                   ▼
             Final Result
