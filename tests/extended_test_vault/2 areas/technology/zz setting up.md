## General
- Apps
	- Chrome/Arc
	- Thunderbird
	- Obsidian
	- Warp
	- PyCharm
	- Fluidvoice
	- Raycast
	- Hidden Bar (menu bar icon manager)
	- Rectangle (window manager)
	- ItsyCal (calendar widget)
	- NotionCalendar
	- SublimeText
	- Warp shell
	- USB Overdrive (mouse configuration)
	- AltTab
	- KarabinerElements
- Comandline tools
	- zoxide (smart cd)
	- bottom (modern htop)
	- eza (modern ls)
	- starship (prompt)
	- ast-grep (grep for agents)
	- uv (package management)
- Fonts
	- NerdFonts (https://www.nerdfonts.com/)
	- Monospace (https://github.com/githubnext/monaspace#monaspace)

If using Brewfile, install via: `brew bundle install --file=~/Brewfile`
```
brew "ast-grep"
brew "bottom"
brew "eza"
brew "starship"
brew "terminal-notifier"
brew "zoxide"
cask "alt-tab"
cask "arc"
cask "freelens"
cask "hiddenbar"
cask "karabiner-elements"
cask "notion-calendar"
cask "obsidian"
cask "raycast"
cask "rectangle"
cask "spotify"
cask "sublime-text"
cask "thunderbird"
cask "warp"
uv "pre-commit"
```
## PyCharm
- use Xenon font from: https://monaspace.githubnext.com/
## Python
On a Mac, use numpy compiled with blas for mac:
```bash
brew install llvm
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export AR=/opt/homebrew/opt/llvm/bin/llvm-ar # (only if on python 3.11)
uv sync --no-binary-package numpy
```

---
Associations:
Tags: #📥️ (needs to be extended with tools I started using)
