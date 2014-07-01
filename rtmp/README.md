RTMP player with delay
===

## Overview

You might be watching World Cup 2014 via sopcast, and listening to the commentators on [VOV3](http://vov3.vov.vn)?
You noticed that the voice (VOV3) always comes before the sopcast signal by several minutes?
Then this script is created for you.
It can playback the VOV3 radio stream with some delay in time, so that it is in sync with whatever you like.

## Usage

In a terminal, run
```sh
$ python rtmpDelay.py 39.5
```
to delay the signal by 39.5 seconds.

You might also want to run
```sh
$ python rtmpDelayThreads.py 39.5
```
for the same purpose. `rtmpDelayThreads.py` does exactly the same thing as `rtmpDelay.py`,
however it is implemented as a multithreaded program, which might be more efficiently.
 
## Prerequisites

It is required that [rtmpdump](http://rtmpdump.mplayerhq.hu/) and [VLC](http://www.videolan.org/vlc/index.html) are installed (`sudo apt-get install rtmpdump vlc`).

The script works as long as you can launch the following command in a terminal (and hear something):

```sh
$ rtmpdump -r rtmp://210.245.60.242:1935/vov3 --playpath=vov3 | vlc -vvv -
```

The Python script is just a wrapper around this command.

Other media player might also work with FLV stream, but I haven't tried. Since the FLV stream is encoded in MP4 format, which is proprietary, VLC might be the best choice.
