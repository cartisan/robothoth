- Close the Loop
- Ralph Wiggum
- Use CLIs instead of MCP
- Safety mantra: ONLY 2 OUT OF 3  
  1. Process Untrustworthy Inputs: Search the web, read emails, etc.  
  2. Access Sensitive Data/Systems:  Read emails, read Notion data  
  3. Change State/Communicate Externally: Write Emails, Execute Code, Change Notion
## Talk Tural
- director of engineering at soundcloud
- Cursor
	- read the doc, it explains many features
	- has rules
		- are added to each prompt -- its a directory with .mdc files each describing a rule
	- has skills
		- like rules, but are only added to prompt when relevant to context
		- can also attach scripts to a skill, by mentioning it in the .mdc and how to use it
	- has agents
		- like skills, but can run in parallel and do not require/take context
			- e.g. code review or security review, subagent should not be biased by context of generation
	- has MCP
		- connections to outside tools
	- has hooks
- general guidelines:
	- Use stronger models for planning, and weaker models for implementation
		- weaker models are still good at following plans
	- Add verbosity rule: 
		- Principle: Everything will be read and reviewed by humans, respect their time
		- Comment: prefer inline comments
- prompts:
	- go from generic to specific
		- principles, specific ideas, examples
		- working on individual features, start a new session for each feature
	- switch to a new context when you are stuck and keep asking the same thing without result
- organizational
	- limit size of MRs to humanly handleable
	- refactoring goes into its own MR
	- agentic review - first pass of review is done by agent
		- capture the style feedback, so that humans can focus on content

##
Associations: [[LLM Prompt Archive]]
Tags: #🌱 
Sources:
