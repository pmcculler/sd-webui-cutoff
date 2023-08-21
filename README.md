# Cutoff - Cutting Off Prompt Effect

## What is this?

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which limits the tokens' influence scope.

## It is also a fork of webui-cutoff

This is a fork of [sd-webui-cutoff](https://github.com/hnmr293/sd-webui-cutoff) with a couple of usability improvements.

* About a hundred single color words will be automatically included as target words if and where they occur in your prompt.
* * Just trying to control basic color bleed? Install the extension, enable it, and you're done. No special target necessary.
  * _Note_ that the original webui-cutoff isn't perfect and this isn't _better_ at what webui-cutoff does. It is _better than nothing_ but don't expect perfection from either extension, please.
  * Webui-cutoff seems to shine best when used to control clothes colors on people with an anime checkpoint. It still works with realistic checkpoints, but perhaps not as reliably.
* You don't have to use the separate target prompt box, and can define your targets inline in the code. So instead of saying "a red apple branch," in your prompt and "red" in your prompt, you can say "a &&red&& apple branch" and skip the target box. Note that targeting a word once targets it everywhere in the prompt, just like using the target prompt box; "a &&red&& apple branch, a boy with a red hat" targets _both_ the apple branch clause and the boy-hat clause.
* That should do it. All the other options and such are unchanged.
