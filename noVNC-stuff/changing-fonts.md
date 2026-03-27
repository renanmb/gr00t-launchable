## Notes on Changing Fonts

So there are several types of VNC server we  can use

THere is noVNC, TigerVNC, TightVNC. They all have different fonts and UI

There are some weird issues with fonts on noVNC. Fonts can be added and changed on Linux easily.

Example:

```bash
sudo apt install fonts-dejavu-core fonts-ubuntu xfonts-base
```

There are several: fonts-noto fonts-liberation2 fonts-fira-sans fonts-hack-ttf ttf-mscorefonts-installer ....

If applications do not immediately recognize the new fonts, rebuild the font cache. This step forces the system to scan all font directories and update its internal font index:

```bash
sudo fc-cache -fv
```

After installation, verify that the fonts are available on your system using the ```fc-list``` command combined with ```grep``` to filter results:

so if you install the microsoft fonts:

```bash
sudo apt install ttf-mscorefonts-installer
```

This will check if the files are installed:

```bash
fc-list | grep -iE "arial|times|verdana|georgia|trebuchet"
```

if it worked you will see:

```bash
/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf: Arial:style=Bold
/usr/share/fonts/truetype/msttcorefonts/Arial.ttf: Arial:style=Regular
/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf: Times New Roman:style=Regular
/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf: Verdana:style=Regular
/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf: Georgia:style=Regular
/usr/share/fonts/truetype/msttcorefonts/trebuc.ttf: Trebuchet MS:style=Regular
```

The package ```ttf-mscorefonts-installer``` is pretty cool, it includes the following fonts:

* Andale Mono: a monospace font for technical documents and code
* Arial (including Arial Black and Bold variants): the standard sans-serif font for business documents
* Comic Sans MS: an informal sans-serif font
* Courier New: a monospace font commonly used in code and formal documents
* Georgia: a serif font optimized for screen readability
* Impact: a heavy sans-serif display font for headlines
* Times New Roman: the standard serif font for academic papers and formal documents
* Trebuchet MS: a sans-serif font popular in presentations
* Verdana: a sans-serif font designed for screen readability
* Webdings: a symbol font for web graphics and decorative elements