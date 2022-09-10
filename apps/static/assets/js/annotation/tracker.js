"use strict";

const binding = require('./dist/binding-browser');
let lib;
let videoKeyInHeap;
let cachedArray;
let canvas;
const videoKey = (video, pointer) => pointer + video.src + video.currentTime;
function videoToArray2D(video, pointer) {
    const key = videoKey(video, pointer);
    if (key === videoKeyInHeap && cachedArray)
        return cachedArray;
    if (!canvas)
        canvas = document.createElement('canvas');
    const { videoWidth: width, videoHeight: height } = video;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    lib.HEAP8.set(pixels, pointer);
    videoKeyInHeap = key;
    cachedArray = lib.readImageData(pointer, width, height);
    return cachedArray;
}

function canvasToArray2D(canvas) {
        const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    lib.HEAP8.set(pixels, pointer);
    videoKeyInHeap = key;
    cachedArray = lib.readImageData(pointer, width, height);
    return cachedArray;
}

class VideoCorrelationTracker {
    static freeMemory() {
        if (!lib)
            return;
        if (this.ptr) {
            lib._free(this.ptr);
            this.ptr = undefined;
        }
    }
    static reserveMemory(video) {
        if (this.ptr)
            this.freeMemory();
        const makePtr = () => this.ptr = lib._malloc(video.videoWidth * video.videoHeight * 4);
        if (!lib) {
            lib = binding();
            lib.then(makePtr);
        }
        else {
            makePtr();
        }
    }
    get videoArray2D() {
        if (!VideoCorrelationTracker.ptr)
            VideoCorrelationTracker.reserveMemory(this.video);
        return videoToArray2D(this.video, VideoCorrelationTracker.ptr);
    }
    get prediction() {
        if (!this.tracker)
            return { x: 0, y: 0, width: 0, height: 0 };
        const rect = this.tracker.predict(this.videoArray2D);
        return {
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height,
        };
    }
    constructor(video, rect) {
        this.video = video;
        this.rect = rect;
        const loadWithLib = () => {
            if (!VideoCorrelationTracker.ptr)
                VideoCorrelationTracker.reserveMemory(video);
            const _rect = new lib.Rectangle(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
            this.tracker = new lib.CorrelationTracker();
            this.tracker.startTrack(this.videoArray2D, _rect);
        };
        if (!lib) {
            lib = binding();
            lib.then(loadWithLib);
        }
        else {
            loadWithLib();
        }
    }
    update(rect) {
        if (!this.tracker)
            return;
        const _rect = new lib.Rectangle(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
        this.tracker.update(this.videoArray2D, _rect);
    }
}
exports.VideoCorrelationTracker = VideoCorrelationTracker;
