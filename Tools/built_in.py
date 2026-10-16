from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search_tool = DuckDuckGoSearchRun()
shell_tool = ShellTool()

result_1 = search_tool.invoke('today news')
result_2 = shell_tool.invoke('whoami')
print(result_1)
print(result_2)

