- Note taking app with linking and back linking capabilities ^bfaa79
##### Content
- [[#Syntax|Syntax]]
- [[#Hotkeys|Hotkeys]]
- [[#Personal Conventions|Personal Conventions]]
- [[#MOCs|MOCs]]
- [[#Plugins|Plugins]]
- [[#Graph view|Graph view]]
- [[#Command line hacks|Command line hacks]]
- [[#Ressources|Ressources]]
## Syntax

| what                   | how                              |
| ---------------------- | -------------------------------- |
| resizing images        | `![[image.png\|150x250]]`        |
| centering images       | `![[image.png\|center]]`         |
| adjust embedding style | `![[page name\|clean no-h ...]]` |
| highlighting           | `==text==`                       |
| external links         | `[text](url)`                    |

## Hotkeys

| what                | how            |
| ------------------- | -------------- |
| go to last/next tab | Alt + <-/->    |
| navigate back       | Alt + z        |
| navigate forward    | Alt + r        |
| Select note         | Ctrl + o       |
| Select command      | Ctrl + p       |
| Close tab           | Ctrl + w       |
| Undo                | ctrl + z       |
| Redo                | ctrl + shift z |
| toggle reading mode | ctrl + e       |
| insert template     | cmd + t        |
| go to weekly note   | cmd + w        |

## Personal Conventions
### Tags
- 📥️: new note that requires processing and expanding, usually in inbox
- 🌱: fresh note that has useful content, but requires proper linking with other notes
- 🌲: fully formed permanent note containing one idea
- 🗺️: MOC or index
- 🔭: overview note, containing several connected ideas or topics

## MOCs
- Sometimes we want to include an overview note in a MOC, such that the headers of this note become part of the MOC (see e.g [[z> MOC Infrastructure#Apps]])
	- Go to the overview note and create a TOC using the TOC plugin
	- Transclude the TOC in the MOC
	- use embedd-adjustment CSS to adjust formatting: `[[note#Content|no-h clean]]`

## Plugins
- **Advanced Tables**: Pane for advanced table editing options
- **Advanced URI**: Allows you to control many different features in Obsidian just by opening some URIs. 
- **Dataview**: query language for notes, allows e.g. finding all open todo notes in a folder
- **Editing Toolbar**: adds toolbar for markdown text editing at top
- **Excalidraw**: advanced canvas
- **Iconize**: allows adding icons to your nacigation pane entries
- **Image Layouts:** Allows displaying images side by side, using the following syntax:![[Pasted image 20241206110535.png]]
- **Image Toolkit:** Enlarges images on click in edit mode, allows zooming
- **Natural Language Dates**: allows turning things like `@Today`into a date
- **Note Refactor**: allows extracting paragraphs into their own notes
- **Outliner**: improves handling of lists
- **Periodic Notes**: allows quarterly and weekly notes to be created, also using templates
- **Reveal Active File Button**: shows active file in navigation pane
- **Show current file path**: shows path of current note at bottom
- **Style settings**: allows changing settings of current style
- **Table fo Contents**: allows automatic ToC creation, useful for overview notes
- **Tag Wramgler**: allows advanced interaction with tags
## Graph view
- Remove weekly notes and templates from graph: `-path:"default_vault/3 resources/1 weekly" -path:"default_vault/3 resources/2 quarterly" -path:templates`

## Command line hacks
### Replace all old_text with new_text: 
```bash
grep -rl 'old-text' . | xargs -d '\n' sed -i 's/old-text/new-text/g'
```
- grep finds all files containing the old-text, recursively, and returns them line separated
- xarg chokes on file names with spaces, `-d` tells it to split on new lines
- it applies the sed command on each line
- sed is set to inplace mode via `-i`
- **danger:** this can fuck up your files irreversibly ;)

## Ressources
- A thread on [Common CSS Hacks](https://forum.obsidian.md/t/meta-post-common-css-hacks/1978/40)
- A guide for the [embedd adjustements CSS](https://github.com/SlRvb/Obsidian--ITS-Theme/blob/main/Guide/Embed-Adjustments.md)


---
Associations:
Tags: #🌲
