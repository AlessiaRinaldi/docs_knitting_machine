# Files ready for the Quarto documentation repository

This folder contains a complete proposed set of source pages and the updated `_quarto.yml` navigation file.

## Replace existing files

Copy these files over the current versions:

```text
_quarto.yml
index.qmd
components.qmd
control-system.qmd
display-interface.qmd
error-results.qmd
error-future_work.qmd
about.qmd
```

The package also includes the current Error Recognition source pages copied from the public repository so that all links remain locally resolvable:

```text
error-recognition.qmd
error-camera.qmd
error-mount.qmd
error-code.qmd
```

## Add new files

```text
pattern-editor.qmd
operating-machine.qmd
raspberry-control.qmd
esp32-actuation.qmd
stepper-motor.qmd
servo-arms.qmd
wiring-power.qmd
troubleshooting.qmd
future-developments.qmd
```

## Add image folders

The empty folders indicate where to place future photographs and renders:

```text
images/display/
images/servo_arms/
images/wiring/
```

Git does not store empty folders; create them again locally when you add the images.

## Preview and publish

From the repository root:

```bash
quarto preview
```

After checking the site:

```bash
quarto render
git add .
git commit -m "Reorganize documentation and add control system pages"
git push
```
