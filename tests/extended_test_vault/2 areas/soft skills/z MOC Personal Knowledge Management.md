## Questions
- What to use tags for?
- When to link notes and how?
- What to do with weekly/quarterly notes?
	- link them to content notes?
- What to do with meeting notes?
- When to use transclusion?

- I like using longer notes that bring together status quo, findings, ideas. How to work with those?
	- Have them as MOCs, that do not only link but also contain summaries?

- Should there be a difference between Knowledge Pages e.g. about a tool, and more Idea Related pages?

## Findings
- When refactoring notes to PARA setup, I found several long notes from lectures that contained knowledge I never looked at again, and that was not connected to anything
	- Gave them a tag of to be reviewed, to extract the useful knowledge
- Workshop notes are also `Literature notes`
	- they need to be processed and turned into `Permanent notes` after the workshop
	- they should go to the archive once the workshop is done

## Concepts
- MOC: map of content
	- like an index
	- central page containing links to related notes (or sub-MOCs)
	- creates nice central hubs in graph view

- PARA:
	- projects: concrete pieces of work that have a start and an end date - have a definition of done
	- areas of interest: general sphere of activity with quality standard - are maintained over time
	- resources: things like templates, meeting notes, zettelkasten; but als topics of interest without standard to be maintained
	- archive: inactive items for all three other groups

- Zettelkasten principle
	- three type of notes
		- Fleeting notes
		- Literature notes
		- Permanent notes
	- goal is to turn fleeting and literature notes into permanent ones
	- each permanent note contains one idea and should be self contained
	- rephrase ideas you find somewhere in your own words
	- always link notes to other notes, otherwise they will get lost
		- Don’t tag notes in the context you found them. Tag them in the context in which you want to discover them.

- Evergreen notes (*Needs to be researched*):
	- Notes should be about one and one thing only (like seperation of concerns in software): 
		- allows for easier linking
		- easier to find
	- Organization of notes should be topical/conceptual (not by person, book, event, ...)
		- allows discovering connections across books, people, etc.
	- Notes should be densely linked
		- pushes us to think how ideas are connected
		- forces to read old notes and revisit ideas
		- should result in surprises when following links
	- Let structure emerge organically
		- imposing it from start results in premature constraining of thought
		- hierarchical ordering is a model, does not reflect nature of things

- Sources
	- https://notes.andymatuschak.org/Evergreen_notes
	- https://stephanango.com/evergreen-notes
	- https://engineeringideas.substack.com/p/reflection-on-two-years-of-writing

- tags
	- two types of tags
		- Tags for topics. You use tags to group notes under a topic
		- Tags for objects. You use tags to group notes around an object, real or conceptual
	- Topic tags get messy really fast because they bring up a lot of unstructured noise
	- Object tags are more useful, since they reference all information for one specofic piece of knowledge. But, then you end up with a loooot of tags

## Workflows
### Joshua Duffney
https://www.youtube.com/watch?v=o_pq43WzeEo&ab_channel=JoshuaDuffney

- Fleeting notes
	- use daily notes to take down anything coming up during the day
	- reduce friction: do not think where to put note, just dump in the daily
	- end of day: review daily note (*don't think this works well for me)
		- boring information kept as text
		- worthwhile ideas should be turned into permanent note and linked to from the daily note
- Permanent notes
	- contain ideas of value
	- all located in one folder ("slip box")
	- naming uses hierarchical prefix based on topic, so that related notes appear below each other in outline (**how is that different from folders and subfolders, just without collapsible structure?!**)![[Pasted image 20221222142834.png|150]]
	- categories are MOCs
- tags: used not to organize knowledge, but to organize the PKM process
	- e.g. tag notes that need reviewing

### Siv WUK
https://www.youtube.com/watch?v=s0OGm5fB4xk&ab_channel=SivW.U.K.

- create source notes on stuff they read
- turn them into permanent notes
- Source notes
	- use topic tags (but apparently also different vaults for different topics)
	- use link to identify type of content
- Permanent notes
	- start and enrichemenet of ideas
	- linking with tags and other ideas
- Block references:
	- use `^status` for a reference named "status" to the block before the caret, then you can link to it  using `![[name#^status]]`  <- this is transclusion
	- You can use this to dynamically import the status of a note e.g. in a table

### Systematic Mastery
https://www.youtube.com/watch?v=t3Qr_rFfPc8&ab_channel=SystematicMastery

- focus on templates
- for each project have a main project note to which all relevant notes are are linked
	- have a daily notes project note, to which all daily notes link
	- have a weekly note which is a retrospection, and link all relevant daily notes to it
- use templater plugin to create templates
- use workbench plugin to keep track of WIP notes

https://www.youtube.com/watch?v=uqVx22lo9_4&ab_channel=SystematicMastery

- uses PARA as main folder structure
	- put most of information into `ressources`
	- Have a `Wiki` for fixed and external information, and a `Zettelkasten` only for ideation
- uses `Tags` and `Associations` in header for extra structure
- use an inbox, where notes end up in per default
- use Ctrl+P and Ctrl+O to access command palette and quick switcher 

### Go paperless
https://go-paperless.net/2021/03/23/notizen-organisieren-in-obsidian-mein-workflow/

- Mix of tags, folders and MOCs
- MOCs
	- All MOCs are in one special folder
	- Has an Index that links to all MOCs
	- MOCs represent Areas of Responsibility
	- MOCs sometimes link to Notes and Project MOCs
		- Project MOCs emerge on the fly from work
		- When they need a note again and again, it turns into a Project MOC
- Uses theme pages instead of tags to represent topics
	- connect to these pages via links
	- better than tags, because theme pages  can be turned into proper notes --> **how do they differ from MOCs?**


### Bryan Jenks
https://www.youtube.com/watch?v=Ewhfok91AdE&ab_channel=BryanJenks

- mass editing notes: not possible in GUI but since its all files you can just script it
- fuzzy search
- always has graph view open in a little box on the right
- uses Andy Mode to lay out cards next to each other
- raindrop.io
	- like a mail box for articles to read on  the internet, including Chrome Extension
	- solves Medium paywall problem
	- can do Stack Overflow, Github, ...
- Literature Notes
	- Live in seedbox state
	- After review, ideas are extracted into seedling notes
	- Incubation phase enriches seedlings with thoughts and links
	- Planting seedlings into evergreen forest: include it into context where ==I want to find it== via links
- Also keeps technical notes, not just ideas
	- e.g. has a Statistics MOC that links to statistics terms and their definitions
- Structure should emerge over time
	- uses links instead of tags to group themes, some of these tag links are notes that do not yet exist
	- finds central nodes that are linked a lot (even when they do not exist) and makes them into MOCs
	- MOCs have a three digit index number (**why?**)
- tags are singles emojis
	- emojis for aesthetics, easier to type?
	- seedbox, seedlings, evergreen, but also: tag-tag for tag-notes, gear for process stuff and chemark for TODO
	- have them on the templates, but have saved searches that remove template notes
	- only have few tags, so they don't get lost
- Prefixes in file names
	- indicates type of content, like Video, ^ for Research Paper, @ for People
	- helps easily filter content type via Obsidian search
	- might cause problems with bash script processing 
- Daily notes
	- link to literature notes created on that day, which link to idea notes created from those lit notes
- *you can embed content in obsidian notes via iframes using html syntax --> have a template for that*
- ==Don't focus on the graphs, the graph is not the goal. The goal is idea synthesis via links. The graph just visualizes this.==


### Denise Todd
https://denisetodd.medium.com/8-features-to-help-you-decide-how-to-use-obsidian-tags-and-links-effectively-6ecc4408d226
- has an extensive comparison of tags vs. links vs. folders
---
Associations: [[self organization]]
Tags: #🗺️ #📥️ (needs to be processed to extract ideas that are collected at the top atm)
