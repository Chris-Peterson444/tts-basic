# tts-basic

A collection of Text-to-Speech (TTS) engine and audio client wrappers to
allow quickly writing a TTS program in Python which can output the audio to
a virtual microphone.

# Why?

I had to go on vocal rest for a few days and charades wasn't working! I also
wanted to come up with a programtic way to interface with my audio systems.
Creating virtual audio devices and connecting them properly usually required
a few different commands that I seldom used - meaning I usually to look it up
everytime - and required using a separate program to manage the audio graph.
Furthermore, the
Instead, I wanted to have a single application I could run that would create
the necessary devices, connect them up as needed, and then tear it down on
completion.

# Usage


## Linux


## Windows and MacOS


See [examples.py](examples.py) for

# Text-to-Speech
Checkout [coqui-ai](https://github.com/coqui-ai/TTS) for the TTS info. It seems
to be the best ML-based TTS library I could find. In particular I found the
VITS model to perform the best. The model is small (~500mb) and works very fast
on GPU and quite fast (<1s) on CPU.

# Audio routing

The hardest part in my opinion. Right now this works for systems running
PulseAudio compatible servers. In my case (Ubuntu 22.04) I'm running PipeWire
which has PA compatibility (via `pipewire-pulse`). `pactl` is missing
modern documentation and there seems to be a lack of guides for how audio
systems in linux work in the first place! For routing one could use any graph
software (e.g., qjackctl, carla, helvum), but that still doesn't solve the
problem of creating the sources and sinks in the first place. Additionally,
I would much prefer to automate this process (i.e. only using CLI tools or APIs).
So this guide is at the very least for a future version of me looking to
work with PA.

One of the biggest questions I find online is how to create a virtual
microphone on Linux. Usually the idea is to route audio output (e.g., from a
browser or game) to another audio input (e.g., into a VOIP client or recording
software). This is often confused with the desire to create a "loopback".
In my experience, a loopback commonly refers to a sink which outputs another
source (usually from some physical device). For example,
you might want to test your microphone and listen to its
output directly. For this, you might create a loopback (a sink) which listens to
your microphone source. Now, a virtual microphone is a reverse-loopback. We
want a source which will pass along the data sent to a sink.

The simplest way to do this is to create a "null" sink with the
`module-null-sink` module. This is
because a sink will (usually; always in the following `pactl` example) have a
corresponding "monitor" source which which reflects the audio sent to the sink.
For example:

```bash
pactl load-module module-null-sink sink_name=example_sink
```

will produce a *sink* named `example_sink` and a *source* named
`example_sink.monitor`. So now you can route audio to `example_sink` and you
can record it as a separate stream from `example_sink.monitor`.

Unfortunatley, we aren't quite done yet. If you needed to pull from this source
programatically you could find it, but for some reason `.monitor` sources
don't show up in applications. You could force the application to use this
source by setting the `PULSE_SOURCE` environment variable when launching
the application (e.g., `PULSE_SOURCE=example_sink.monitor discord`) but this can be
ignored and is a little clunky for regular usage. See the PA wiki for
more information on [accepted environment variables](https://www.freedesktop.org/wiki/Software/PulseAudio/FAQ/#whatenvironmentvariablesdoespulseaudiocareabout).

So we have to make a proper audio *source*. To do this we can abuse the
`module-remap-source` module. This module creates a virtual source
on top of another source to allow for shuffling the channel map. We can use
this module to simply create a virtual source our applications should see.
Continuing off the previous example:
```bash
pactl load-module module-remap-source source_name=TTSMic master=example_sink.monitor
```

Now in your application of choice, we'll see a `TTSMic` input which will play
anything you send to `example_sink`.



## Extras

You can find (most of) the available modules to `pactl` on the PA wiki
[modules page](https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/Modules/).

Although this page doesn't seem to always be up to date. On my system there are
more modules available to me than listed here. It may be easier to inspect
the [PulseAudio source code](https://gitlab.freedesktop.org/pulseaudio/pulseaudio/-/blob/master/src/modules/)
and some trial and error to determine which modules may be available.

Unfortunately there does not seem to be a way to inspect which modules are
available to your installed version of `pactl` (why??).

In my search, I found the modules `module-virtual-source` and `module-null-sink`
available to me. It appears that `module-virtual-source`'s behavior is a
superset of `module-remap-source`. So a similar command might be:
```bash
pactl load-module module-virtual-source source_name=TTSMic master=example_sink.monitor
```

I think I prefer this way (it shows up a little nicer by default and it also
makes more sense), but this is unfortunately undocumented. One caveat with this
module is that it's meant to filter another source (or alternatively a sink for
`module-virtual-sink`) so you *have* to specify a master source, otherwise
the system default source will be used (which is probably not what you want!).


