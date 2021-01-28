(function() {
document.onload = initialize();

// ----------------------------------------------------------------------------------------------------
// init, dialog, icons
// ----------------------------------------------------------------------------------------------------

function initialize() {
    console.log('initializing');

    htmlCanvas = document.getElementById('c'),
    context = htmlCanvas.getContext('2d');

    htmlCanvas.addEventListener('touchstart',touchStart,false);
    htmlCanvas.addEventListener('touchend',touchEnd,false);
    htmlCanvas.addEventListener('touchcancel',touchCancel,false);
    htmlCanvas.addEventListener('touchmove',touchMove,false);

    // 'https' does not work on Firefox (it's ok though since http redirects to https)
    NS = 'http://www.w3.org/2000/svg';
    svg = document.createElementNS(NS,'svg');
    svgDiv = document.getElementById('s');
    svgDiv.appendChild(svg);
    // also using http://svgjs.com/

    dialogArea = document.getElementById('d');
    textInput = document.getElementById('i');
    iconsArea = document.getElementById('icons');
    iconPointer = document.getElementById('icon-pointer');
    iconZoomIn = document.getElementById('icon-zoomin');
    iconZoomOut = document.getElementById('icon-zoomout');
    iconMove = document.getElementById('icon-move');
    currentStateIcon = iconMove;
    mouseIsDown = false;
    canvasWidthDiffToWindow = 300;
    canvasHeightDiffToWindow = 25;
    canvasCenter = null; // x (col), y (row)
    mouseDownLoc = null;

    window.addEventListener('resize', resizeHTMLCanvas, false);
    htmlCanvas.addEventListener('mousemove', mouseMove);
    htmlCanvas.addEventListener('mousedown', mouseDown);
    htmlCanvas.addEventListener('mouseup', mouseUp);
    textInput.addEventListener('keydown',keyDown);
    iconPointer.addEventListener('mousedown',iconClicked);
    iconZoomIn.addEventListener('mousedown',iconClicked);
    iconZoomOut.addEventListener('mousedown',iconClicked);
    iconMove.addEventListener('mousedown',iconClicked);
    textInput.value = ''

    // SocketIO
    socket = io.connect('http://' + document.domain + ':' + location.port);
    // socket = io.connect('https://' + document.domain + ':' + location.port, {secure: true});
    // socket = io.connect('http://localhost:5000/');
    socket.on('connect', function() {
        console.log('Client: SocketIO connected');
    });
    socket.on('disconnect', function() {
        console.log('SocketIO disconnected');
    });
    socket.on('server2ClientMessage', function(data) {
        // console.log(data);
        if (data == 'server initialized') {
            addMessageToDialog('ready | enter instructions above','right');
        } else if (data == 'label mask created') { // show annotation tool
            annotateStep = 2; showTool(); enableIconPointer();
        } else if (data == 'done saving annotations') {
            dismissAnnotationTool();
        } else if (data == 'done interpolating plane masks') {
            addMessageToDialog('done interpolating annotations','right');
            showTool(); enableIconPointer();
        } else if (data == 'animate last dialog message') {
            animateLastDialogMessage();
        } else if (data == 'did annotate images') {
            didAnnotateImages = true;
        } else if (data == 'did train ml segmenter') {
            didTrainMLSegmenter = true;
        } else if (data == 'fetch plane') {
            fetchPlane();
        } else if (data == 'fetch image') {
            fileName = 'Image.tif';
            fetchFile('fetchim');
        } else if (data == 'fetch data') {
            fileName = 'Annotations.zip';
            fetchFile('fetchdt');
        } else if (data == 'fetch table') {
            fileName = 'Table.csv';
            fetchFile('fetchtb');
        } else if (data.slice(0,8) == 'nclasses') {
            nClasses = parseInt(data.slice(9));
        } else if (data == 'load annotation tool') {
            toolAnnotate('1');
        } else if (data == 'load new image tool') {
            toolNewImage();
        } else if (data == 'load find spots tool') {
            toolFindSpots();
        }
    });
    socket.on('dialog', function(data) {
        addMessageToDialog(data,'right');
        if (data == 'done training') {
            didTrainMLSegmenter = true;
        }
    });
    socket.emit('client2ServerMessage', 'socket echo test');

    // http://flask.pocoo.org/docs/0.12/patterns/jquery/
    $.getJSON($SCRIPT_ROOT + '/jsonecho', {message: 'json echo test'}, function(data) { console.log(data.message); });

    onMobile = false;
    if( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
        onMobile = true;
    }
    if (!onMobile) { // no auto-complete on mobile
        var availableTags = ['new image', 'dismiss tool', 'clear mask', 'clear dialog', 'image properties',
                             'annotate', 'annotate with interpolation', 'train ml segmenter', 'segment', 'new image from server', 'reset image',
                             'blur', 'log', 'gradient magnitude', 'derivatives', 'ml probability maps', 'save current image to server',
                             'save ml model to server', 'load ml model from server', 'extract plane', 'extract channel',
                             'median filter', 'maximum filter', 'minimum filter', 'save annotations to server', 'load annotations from server',
                             'edit annotations', 'download current image', 'download annotations', 'find spots', 'download table'];
        $("#i").autocomplete({source: availableTags});
        $("#i").on("autocompleteclose", function(event, ui) {
            $("#i").val('');
        });
    }

    tool = null;

    imZoom = 1.0;
    image = null;
    imageCanvas = null;
    maskCanvas = null;
    labelMask = null;
    labelMaskForInterp = null;
    nPlanes = nChannels = 0;
    annotateWithInterpolation = false;
    setDefaultImage();

    scheduledPlaneIndex = [0,0,0]; // used during annotation
    scheduledPlaneView = 'z' // used during annotation
    planeView = null;
    annotateStep = segmentStep = 0; // for dialog logic
    didAnnotateImages = false; // ml segmenter needs to know this if called
    didSetImage = false; // otherwise won't let user annotate (because server doesn't have any image)
    didTrainMLSegmenter = false; // segmenter needs to know this if called
    planeSliderPresent = channelSliderPresent = false;
    
    thrButton = document.createElement('INPUT');
    thrButton.setAttribute('type','button');
    $(thrButton).button();
    thrButton.addEventListener('click', switchThr, false);
    thrButton.style.top = 0;
    thrButton.style.left = 180;
    thrButton.style.width = 40;
    thrButton.style.height = iconsArea.clientHeight-2;
    thrButton.style.fontSize = 10;
    thrButton.style.fontWeight = 'bold';
    thrButton.value = '.[...';
    thrButton.style.opacity = 0.1;
    thrButton.disabled = true;
    iconsArea.appendChild(thrButton);

    thrSlider = document.createElement('INPUT');
    thrSlider.setAttribute('type','range');
    thrSlider.setAttribute('name','thrSlider');
    thrSlider.setAttribute('id','thrSlider');
    thrSlider.setAttribute('min','0');
    thrSlider.setAttribute('max','255');
    thrSlider.setAttribute('value','0');
    thrSlider.style.top = 0;
    thrSlider.style.left = 220;
    thrSlider.style.width = 100;
    thrSlider.style.height = iconsArea.clientHeight-2;
    thrSlider.oninput = function () { changeDisplayThreshold(parseInt(this.value)); }
    thrSlider.style.opacity = 0.1;
    thrSlider.disabled = true;
    iconsArea.appendChild(thrSlider);
    thrLower = 0; thrUpper = 255;

    docButton = document.createElement('INPUT');
    docButton.setAttribute('type','button');
    $(docButton).button();
    docButton.style.top = 0;
    docButton.style.left = window.innerWidth-canvasWidthDiffToWindow-52;
    docButton.style.width = 50;
    docButton.style.height = iconsArea.clientHeight-2;
    docButton.style.fontSize = 10;
    docButton.style.fontWeight = 'bold';
    docButton.value = 'docs';
    docButton.style.opacity = 1;
    $(docButton).click(function() { window.open('/docs/index.html','_blank'); });
    iconsArea.appendChild(docButton);

    disableIconPointer();
    textInput.focus();
    socket.emit('client2ServerMessage', 'initialize'); // response ('server initialized') handled above
    socket.emit('client2ServerMessage', 'client is on mobile: ' + Number(onMobile));
}

function switchThr() {
    if (thrButton.value == '.[...') {
        thrButton.value = '...].';
        thrSlider.value = thrUpper;
    } else {
        thrButton.value = '.[...';
        thrSlider.value = thrLower;
    }
}

function changeDisplayThreshold(value) {
    var lt, ut;
    if (thrButton.value == '.[...') {
        lt = value;
        ut = thrUpper;
    } else {
        lt = thrLower;
        ut = value;
    }
    
    if (lt < ut) {
        setImDisplayThresholds(lt,ut);
        thrLower = lt;
        thrUpper = ut;
    }
}

function setImDisplayThresholds(lt,ut) {
    for (var i = 0; i < imageCanvas.height; i++) {
        for (var j = 0; j < imageCanvas.width; j++) {
            var d = originalImageData[i*imageCanvas.width+j];
            if (d < lt) {
                d = 0;
            } else if (d > ut) {
                d = 255;
            } else {
                d = (d-lt)/(ut-lt)*255;
            }
            var index  = i*imageCanvas.width*4+j*4;
            imageData[index]   = d;
            imageData[index+1] = d;
            imageData[index+2] = d;
            imageData[index+3] = 255;
        }
    }
    imageContext.putImageData(contextImageData,0,0);
    redraw();
}

function disableIconPointer() {
    iconPointer.removeEventListener('mousedown',iconClicked);
    iconPointer.style.color = 'rgb(100,100,100)';
    if (currentStateIcon == iconPointer) {
        iconMove.style.color = 'white';
        currentStateIcon = iconMove;
    }
}

function enableIconPointer() {
    iconPointer.addEventListener('mousedown',iconClicked);
    iconPointer.style.color = 'rgb(200,200,200)';
}

function setDefaultImage() {
    imCenter = null; // for display; x (col), y (row)
    imOrigin = null; // for display; x (col), y (row)
    imDispSize = null; // display size; width, height
    imOriginBeforeMoving = [0,0];

    image = new Image();
    image.addEventListener('load',imageDidLoad,false);
    image.src = '../static/PuBliCiTy.png';
}

function imageDidLoad() {
    imActSize = [image.width, image.height]; // actual size; width, height
    setMaskCanvas();
    resizeHTMLCanvas();
}

function centeredImageCoordinates() {
    imCenter = canvasCenter;
    var c0 = htmlCanvas.width/imActSize[0];
    var c1 = htmlCanvas.height/imActSize[1];
    var z = 0.9*(c0 < c1 ? c0 : c1); // this solves {max zwh subject to zw < 0.9W, zh < 0.9H}, i.e. max area of display image s.t. entire image visible
    imZoom = (z < 1 ? z : 1); // but we're only zooming down from the original (most cases anyway), displaying the original size otherwise
    imDispSize = [Math.round(imZoom*imActSize[0]), Math.round(imZoom*imActSize[1])];
    imOrigin = [Math.round(imCenter[0]-imDispSize[0]/2), Math.round(imCenter[1]-imDispSize[1]/2)];
}

function iconClicked(event) {
    currentStateIcon.style.color = 'rgb(200,200,200)';
    event.target.style.color = 'white';
    currentStateIcon = event.target;
}

function addMessageToDialog(message,handedness) {
    var newParagraph = document.createElement('P');
    newParagraph.className = handedness; // 'left': from client, 'right': from server
    var msg = message;
    if (handedness == 'left') {
        msg = '\u25A1 '+msg;
    } else {
        msg = msg+' \u25A1';
    }
    var newText = document.createTextNode(msg);
    newParagraph.appendChild(newText);
    dialogArea.prepend(newParagraph);
}

function animateLastDialogMessage() { // when more information is needed
    $(dialogArea.firstChild).toggle({effect: 'scale', percent: 102, direction: 'horizontal'});
    $(dialogArea.firstChild).toggle({effect: 'scale', percent: 102, direction: 'horizontal'});
}

// handles dialog commands
function keyDown(event) {
    if (event.keyCode == 13) {
        addMessageToDialog(textInput.value,'left');

        if (textInput.value == 'test1') { // used for testing during development (e.g. draw svg)
            console.log('test1');
        } else if (textInput.value == 'test2') { // used for testing during development (e.g. delete svg)
            console.log('test2');
        } else if (textInput.value == 'fetch plane') {
            // temporary... during development
            // used to fetch result when socket connection is lost, which sometimes happens if processing takes a long time
            fetchPlane();
        } else if (textInput.value == 'clear mask') {
            clearMask();
        } else if (textInput.value == 'new image') {
            annotateStep = segmentStep = 0;
            socket.emit('dialog', textInput.value);
        } else if (textInput.value == 'new image from server') {
            annotateStep = segmentStep = 0;
            socket.emit('dialog', textInput.value);
        } else if (textInput.value == 'dismiss tool') {
            hideTool();
        } else if (textInput.value == 'clear dialog') {
            while (dialogArea.firstChild) {
                dialogArea.removeChild(dialogArea.firstChild);
            }
        } else if (textInput.value == 'annotate') {
            if (didSetImage) {
                addMessageToDialog('how many classes? [1,2,...]','right');
                animateLastDialogMessage();
                annotateStep = 1;
                annotateWithInterpolation = false;
            } else {
                addMessageToDialog('set image before annotating','right');
                animateLastDialogMessage();
            }
        } else if (textInput.value == 'annotate with interpolation') { 
            if (didSetImage) {
                if (nPlanes > 0) {
                    if (nChannels == 0) {
                        addMessageToDialog('how many classes? [1,2,...]','right');
                        animateLastDialogMessage();
                        annotateStep = 1;
                        annotateWithInterpolation = true;
                    } else {
                        addMessageToDialog('image should have a single channel','right');
                        animateLastDialogMessage();    
                    }
                } else {
                    addMessageToDialog('image should be 3D','right');
                    animateLastDialogMessage();    
                }
            } else {
                addMessageToDialog('set image before annotating','right');
                animateLastDialogMessage();
            }
        } else if (textInput.value == 'tmls' || textInput.value == 'train ml segmenter') {
            if (!didAnnotateImages) {
                addMessageToDialog('you need to annotate images first | call \'annotate\' to do so.','right');
                animateLastDialogMessage();
            } else {
                toolTrainMLSegmenter();
            }
        } else if (textInput.value == 'segment') {
            addMessageToDialog('which method? [1] basic threshold; [2] ml segmenter','right');
            animateLastDialogMessage();
            segmentStep = 1;
        } else if (!isNaN(parseInt(textInput.value))) {
            if (annotateStep == 1) { // expects number of classes
                nClasses = parseInt(textInput.value);
                socket.emit('client2ServerMessage', 'nclasses '+nClasses);
                toolAnnotate('0');
            } else if (segmentStep == 1) { // expects type of segmentation
                if (parseInt(textInput.value) == 1) { // basic threshold
                    toolThrSegment();
                } else if (parseInt(textInput.value) == 2) { // ml segmenter
                    if (didTrainMLSegmenter) {
                        toolMLSegment();
                    } else {
                        addMessageToDialog('model unavailable | call \'train ml segmenter\' or \'load ml model from server\'','right');
                        animateLastDialogMessage();
                    }
                }
                segmentStep = 2;
            } else {
                socket.emit('dialog', textInput.value);    
            }
        } else {
            socket.emit('dialog', textInput.value);
        }

        textInput.value = '';
        if (!onMobile) {
            $("#i").autocomplete("close");
        } else {
            $("#i").blur(); // dismiss keyboard
        }
    }
}

// called when plane is generated on server side, and server communicates that's ready to fetch
// e.g. when a new image is set
function fetchPlane() {
    var formdata = new FormData();
    formdata.append('text', JSON.stringify({description: 'fetchpln'}));
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
        success: function(data)
        {
            setPlane(data.tcpi,data.shape,data.data,data.planeView,data.mask,data.maskType,true);
        }
    });
}

// called when client is requesting a new plane
// e.g. when a 3D image is available and user changes the plane slider position
function newPlaneFromServer(tIndex,cIndex,pIndex) { // time, channel, plane (z), mask type ('noMask','labelMask','segmMask')
    var formdata = new FormData();
    formdata.append('text', JSON.stringify({description: 'plnfrmsvr', content: [tIndex,cIndex,pIndex]}));
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
        success: function(data)
        {
            setPlane(data.tcpi,data.shape,data.data,data.planeView,data.mask,data.maskType,false);
        }
    });
}

// called when image is generated on server side, and server communicates that's ready to fetch; triggers download
function fetchFile(msgToServer) {
    var formdata = new FormData();
    formdata.append('text', JSON.stringify({description: msgToServer}));
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false, xhrFields: { responseType: 'blob' },
        success: function(data)
        {
            var link = document.createElement('a');
            link.href = window.URL.createObjectURL(data);
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
}

// ----------------------------------------------------------------------------------------------------
// canvas
// ----------------------------------------------------------------------------------------------------

function movePlaneSlider(yPosition) {
    var w = htmlCanvas.width;
    var h = htmlCanvas.height;
    var hSize = planeSliderHandleCoord[3]-planeSliderHandleCoord[1];
    var y0 = h-1.5*hSize; // bottom
    var y1 = 1.5*hSize; // top
    
    var y = yPosition;
    if (y < y1) y = y1;
    if (y > y0) y = y0;
    var sCentY = y;

    planeSliderHandleCoord[1] = sCentY-hSize/2;
    planeSliderHandleCoord[3] = sCentY+hSize/2;
    planeSliderHandle.setAttribute('y',sCentY-0.9*hSize/2);

    var interval = (y0-y1)/(nPlanes-1);
    planeIndex = Math.round((y0-y)/interval);
    var y = y0-planeIndex*interval;
    planeIndexLine.setAttribute('y1', y);
    planeIndexLine.setAttribute('y2', y);

    if (nChannels == 0) {
        svgTextIndices.text('plane '+(planeIndex+1).toString())
    } else {
        svgTextIndices.text('plane '+(planeIndex+1).toString()+', channel '+(channelIndex+1).toString())
    }
}

function moveChannelSlider(yPosition) {
    var w = htmlCanvas.width;
    var h = htmlCanvas.height;
    var hSize = channelSliderHandleCoord[3]-channelSliderHandleCoord[1];
    var y0 = h-1.5*hSize; // bottom
    var y1 = 1.5*hSize; // top
    
    var y = yPosition;
    if (y < y1) y = y1;
    if (y > y0) y = y0;
    var sCentY = y;

    channelSliderHandleCoord[1] = sCentY-hSize/2;
    channelSliderHandleCoord[3] = sCentY+hSize/2;
    channelSliderHandle.setAttribute('y',sCentY-0.9*hSize/2);

    var interval = (y0-y1)/(nChannels-1);
    channelIndex = Math.round((y0-y)/interval);
    var y = y0-channelIndex*interval;
    channelIndexLine.setAttribute('y1', y);
    channelIndexLine.setAttribute('y2', y);

    svgTextIndices.text('plane '+(planeIndex+1).toString()+', channel '+(channelIndex+1).toString())
}

function snapPlaneSlider() {
    var hSize = planeSliderHandleCoord[3]-planeSliderHandleCoord[1];
    var y = parseInt(planeIndexLine.getAttribute('y1'));
    planeSliderHandleCoord[1] = y-hSize/2;
    planeSliderHandleCoord[3] = y+hSize/2;
    planeSliderHandle.setAttribute('y',y-0.9*hSize/2);

    if (planeIndex != previousPlaneIndex) {
        if (annotateStep == 2) {
            uploadMaskPlane('newPlane');
            scheduledPlaneIndex = [timeIndex, channelIndex, planeIndex];
        } else {
            newPlaneFromServer(timeIndex,channelIndex,planeIndex);
        }
        previousPlaneIndex = planeIndex;
    }
}

function snapChannelSlider() {
    var hSize = channelSliderHandleCoord[3]-channelSliderHandleCoord[1];
    var y = parseInt(channelIndexLine.getAttribute('y1'));
    channelSliderHandleCoord[1] = y-hSize/2;
    channelSliderHandleCoord[3] = y+hSize/2;
    channelSliderHandle.setAttribute('y',y-0.9*hSize/2);

    if (channelIndex != previousChannelIndex) {
        if (annotateStep == 2) { 
            scheduledPlaneIndex = [timeIndex, channelIndex, planeIndex];
            uploadMaskPlane('newPlane');
        } else  {
            newPlaneFromServer(timeIndex,channelIndex,planeIndex); 
        }
        previousChannelIndex = channelIndex;
    }
}

function drawPlaneSlider() {
    planeSliderPresent = true;

    var w = htmlCanvas.width;
    var h = htmlCanvas.height;
    var hSize = 20; // slider handle size
    var y0 = h-1.5*hSize; // bottom
    var y1 = 1.5*hSize; // top
    var interval = (y0-y1)/(nPlanes-1);
    var sCentX = hSize/2;
    var sCentY = y0-planeIndex*interval;

    var svgLine = document.createElementNS(NS,'line');
    var x = sCentX;
    svgLine.setAttribute('x1', x);
    svgLine.setAttribute('y1', y0);
    svgLine.setAttribute('x2', x);
    svgLine.setAttribute('y2', y1);
    svgLine.setAttribute('id','planeSlider');
    svgLine.style.stroke = 'gray';
    svg.appendChild(svgLine);

    for (var i = 0; i < nPlanes; i++) {
        var svgLine = document.createElementNS(NS,'line');
        var y = y0-i*interval;
        svgLine.setAttribute('x1', sCentX-0.5*hSize/2);
        svgLine.setAttribute('y1', y);
        svgLine.setAttribute('x2', sCentX+0.5*hSize/2);
        svgLine.setAttribute('y2', y);
        svgLine.setAttribute('id','planeSlider');
        svgLine.style.stroke = 'gray';
        svg.appendChild(svgLine);
    }
    planeIndexLine = document.createElementNS(NS,'line');
    var y = y0-planeIndex*interval;
    planeIndexLine.setAttribute('x1', sCentX-0.5*hSize/2);
    planeIndexLine.setAttribute('y1', y);
    planeIndexLine.setAttribute('x2', sCentX+0.5*hSize/2);
    planeIndexLine.setAttribute('y2', y);
    planeIndexLine.style.stroke = 'white';
    planeIndexLine.setAttribute('stroke-width',2);
    planeIndexLine.setAttribute('id','planeSlider');
    svg.appendChild(planeIndexLine);

    planeSliderHandleCoord = [sCentX-hSize/2, sCentY-hSize/2, sCentX+hSize/2, sCentY+hSize/2];
    planeSliderHandle = document.createElementNS(NS,'rect');
    planeSliderHandle.setAttribute('x',sCentX-0.8*0.9*hSize/2);
    planeSliderHandle.setAttribute('y',sCentY-0.9*hSize/2);
    planeSliderHandle.setAttribute('width',0.8*0.9*hSize);
    planeSliderHandle.setAttribute('height',0.9*hSize);
    planeSliderHandle.style.stroke = 'white';
    planeSliderHandle.style.fill = 'white';
    planeSliderHandle.setAttribute('fill-opacity',0);
    planeSliderHandle.setAttribute('stroke-width',2);
    planeSliderHandle.setAttribute('id','planeSlider');
    svg.appendChild(planeSliderHandle);

    if (nChannels == 0) {
        svgTextIndices = SVG(svg).text('plane '+(planeIndex+1).toString())
    } else {
        svgTextIndices = SVG(svg).text('plane '+(planeIndex+1).toString()+', channel '+(channelIndex+1).toString())
    }
    svgTextIndices.id('planeSlider').move(25,h-43).font({ fill: 'gray'})

    if (nChannels == 0) { // only draw plane view controls for 3D images
        drawPlaneViewControls();
    }
}

function removePlaneSlider() {
    for (var i = svg.childElementCount-1; i >= 0; i--) {
        if (svg.childNodes[i].id == 'planeSlider') {
            svg.removeChild(svg.childNodes[i]);
        }
    }
    planeSliderPresent = false;

    if (nChannels == 0) { // only draw plane view controls for 3D images
        removePlaneViewControls();
    }
}

function drawPlaneViewControls() {
    pzr = [40,45,20,15]

    svgPlaneOutline = SVG(svg).rect(pzr[2]+17,pzr[3]+17)
    svgPlaneOutline.move(pzr[0]-15,pzr[1]-15).stroke('white').id('planeZ').opacity(0.5)

    svgPlaneZ = SVG(svg).rect(pzr[2],pzr[3])
    svgPlaneZ.move(pzr[0],pzr[1]).fill('gray').id('planeZ').opacity(0.5)
    svgTextPlaneZ = SVG(svg).text('z')
    svgTextPlaneZ.id('svgText').move(pzr[0]+6,pzr[1]-2).font({ fill: 'black'})

    pyr = [pzr[0],pzr[1]-13,pzr[2],12]
    svgPlaneY = SVG(svg).rect(pyr[2],pyr[3])
    svgPlaneY.move(pyr[0],pyr[1]).fill('gray').id('planeY').opacity(0.5)
    svgTextPlaneY = SVG(svg).text('y')
    svgTextPlaneY.id('svgText').move(pyr[0]+6,pyr[1]-5).font({ fill: 'black'})

    pxr = [pzr[0]-13,pzr[1],12,pzr[3]]
    svgPlaneX = SVG(svg).rect(pxr[2],pxr[3])
    svgPlaneX.move(pxr[0],pxr[1]).fill('gray').id('planeX').opacity(0.5)
    svgTextPlaneX = SVG(svg).text('x')
    svgTextPlaneX.id('svgText').move(pxr[0]+3,pxr[1]-2).font({ fill: 'black'})

    if      (planeView == 'z') { svgPlaneZ.fill('white'); }
    else if (planeView == 'y') { svgPlaneY.fill('white'); }
    else if (planeView == 'x') { svgPlaneX.fill('white'); }
}

function removePlaneViewControls() {
    svgPlaneOutline.remove()
    svgPlaneX.remove()
    svgPlaneY.remove()
    svgPlaneZ.remove()
    svgTextPlaneX.remove()
    svgTextPlaneY.remove()
    svgTextPlaneZ.remove()
}

function drawChannelSlider() {
    channelSliderPresent = true;

    var w = htmlCanvas.width;
    var h = htmlCanvas.height;
    var hSize = 20; // slider handle size
    var y0 = h-1.5*hSize; // bottom
    var y1 = 1.5*hSize; // top
    var interval = (y0-y1)/(nChannels-1);
    var sCentX = w-hSize/2;
    var sCentY = y0-channelIndex*interval;

    var svgLine = document.createElementNS(NS,'line');
    var x = sCentX;
    svgLine.setAttribute('x1', x);
    svgLine.setAttribute('y1', y0);
    svgLine.setAttribute('x2', x);
    svgLine.setAttribute('y2', y1);
    svgLine.style.stroke = 'gray';
    svgLine.setAttribute('id', 'channelSlider');
    svg.appendChild(svgLine);

    for (var i = 0; i < nChannels; i++) {
        var svgLine = document.createElementNS(NS,'line');
        var y = y0-i*interval;
        svgLine.setAttribute('x1', sCentX-0.5*hSize/2);
        svgLine.setAttribute('y1', y);
        svgLine.setAttribute('x2', sCentX+0.5*hSize/2);
        svgLine.setAttribute('y2', y);
        svgLine.setAttribute('id','tick');
        svgLine.style.stroke = 'gray';
        svgLine.setAttribute('id', 'channelSlider');
        svg.appendChild(svgLine);
    }
    channelIndexLine = document.createElementNS(NS,'line');
    var y = y0-channelIndex*interval;
    channelIndexLine.setAttribute('x1', sCentX-0.5*hSize/2);
    channelIndexLine.setAttribute('y1', y);
    channelIndexLine.setAttribute('x2', sCentX+0.5*hSize/2);
    channelIndexLine.setAttribute('y2', y);
    channelIndexLine.style.stroke = 'white';
    channelIndexLine.setAttribute('stroke-width',2);
    channelIndexLine.setAttribute('id', 'channelSlider');
    svg.appendChild(channelIndexLine);

    channelSliderHandleCoord = [sCentX-hSize/2, sCentY-hSize/2, sCentX+hSize/2, sCentY+hSize/2];
    channelSliderHandle = document.createElementNS(NS,'rect');
    channelSliderHandle.setAttribute('x',sCentX-0.8*0.9*hSize/2);
    channelSliderHandle.setAttribute('y',sCentY-0.9*hSize/2);
    channelSliderHandle.setAttribute('width',0.8*0.9*hSize);
    channelSliderHandle.setAttribute('height',0.9*hSize);
    channelSliderHandle.style.stroke = 'white';
    channelSliderHandle.style.fill = 'white';
    channelSliderHandle.setAttribute('fill-opacity',0);
    channelSliderHandle.setAttribute('stroke-width',2);
    channelSliderHandle.setAttribute('id', 'channelSlider');
    svg.appendChild(channelSliderHandle);
}

function removeChannelSlider() {
    for (var i = svg.childElementCount-1; i >= 0; i--) {
        if (svg.childNodes[i].id == 'channelSlider') {
            svg.removeChild(svg.childNodes[i]);
        }
    }
    channelSliderPresent = false;
}

function touchStart(evt) {
    // socket.emit('client2ServerMessage', 'touch start');
    evt.preventDefault();
    var event = {layerX: evt.layerX, layerY: evt.layerY};
    mouseDown(event);
}

function touchMove(evt) {
    // socket.emit('client2ServerMessage', 'touch move');
    evt.preventDefault();
    var event = {layerX: evt.layerX, layerY: evt.layerY};
    mouseMove(event);
}

function touchEnd(evt) {
    // socket.emit('client2ServerMessage', 'touch end');
    evt.preventDefault();
    var event = {layerX: evt.layerX, layerY: evt.layerY};
    mouseUp(event);
}

function touchCancel(evt) {
    // socket.emit('client2ServerMessage', 'touch cancel');
    // evt.preventDefault();
}

function mouseDown(event) {
    // socket.emit('client2ServerMessage', 'mouse down '+event.layerX+' '+event.layerY);
    mouseIsDown = true;
    var x = event.layerX;
    var y = event.layerY;
    if (planeSliderPresent && nChannels == 0 && // nChannel == 0 because plane view controls are only present for single channel volumes
        x >= pzr[0] && x <= pzr[0]+pzr[2] &&
        y >= pzr[1] && y <= pzr[1]+pzr[3]) {
        if (annotateStep == 2) { 
            scheduledPlaneView = 'z'
            uploadMaskPlane('newView');
        } else {
            socket.emit('client2ServerMessage', 'view plane z');
        }
        movingPlaneSlider = false; movingChannelSlider = false; mouseIsDown = false;
    } else if (planeSliderPresent && nChannels == 0 &&
        x >= pyr[0] && x <= pyr[0]+pyr[2] &&
        y >= pyr[1] && y <= pyr[1]+pyr[3]) {
        if (annotateStep == 2) {
            scheduledPlaneView = 'y';
            uploadMaskPlane('newView');
        } else {
            socket.emit('client2ServerMessage', 'view plane y');
        }
        movingPlaneSlider = false; movingChannelSlider = false; mouseIsDown = false;
    } else if (planeSliderPresent && nChannels == 0 &&
        x >= pxr[0] && x <= pxr[0]+pxr[2] &&
        y >= pxr[1] && y <= pxr[1]+pxr[3]) {
        if (annotateStep == 2) {
            scheduledPlaneView = 'x';
            uploadMaskPlane('newView');
        } else {
            socket.emit('client2ServerMessage', 'view plane x');
        }
        movingPlaneSlider = false; movingChannelSlider = false; mouseIsDown = false;
    } else if (planeSliderPresent &&
        x >= planeSliderHandleCoord[0] && x <= planeSliderHandleCoord[2] &&
        y >= planeSliderHandleCoord[1] && y <= planeSliderHandleCoord[3]) {
        planeSliderHandle.setAttribute('fill-opacity',0.5);
        movingPlaneSlider = true; movingChannelSlider = false;
    } else if (channelSliderPresent &&
        x >= channelSliderHandleCoord[0] && x <= channelSliderHandleCoord[2] &&
        y >= channelSliderHandleCoord[1] && y <= channelSliderHandleCoord[3]) {
        channelSliderHandle.setAttribute('fill-opacity',0.5);
        movingChannelSlider = true; movingPlaneSlider = false;
    } else {
        mouseDownLoc = [x, y];
        if (currentStateIcon == iconPointer) {
            annotateAtPoint(mouseDownLoc[0],mouseDownLoc[1]);
        } else if (currentStateIcon == iconMove) {
            imOriginBeforeMoving[0] = imOrigin[0]; imOriginBeforeMoving[1] = imOrigin[1];
        }
        movingPlaneSlider = false; movingChannelSlider = false;
    }
}

function mouseUp(event) {
    // socket.emit('client2ServerMessage', 'mouse up');
    mouseIsDown = false;
    if (movingPlaneSlider) {
        snapPlaneSlider();
        planeSliderHandle.setAttribute('fill-opacity',0);
    } else if (movingChannelSlider) {
        snapChannelSlider();
        channelSliderHandle.setAttribute('fill-opacity',0);
    } else {
        var x = event.layerX;
        var y = event.layerY;
        if (currentStateIcon ==  iconZoomIn || currentStateIcon == iconZoomOut) {
            if (x >= imOrigin[0] && x <= imOrigin[0]+imDispSize[0] && y >= imOrigin[1] && y <= imOrigin[1]+imDispSize[1]) {
                var factor;
                if (currentStateIcon == iconZoomIn) {
                    factor = 2;
                } else if (currentStateIcon == iconZoomOut) {
                    factor = 0.5
                }
                imZoom = factor*imZoom;
                imOrigin[0] = Math.round(x-factor*(x-imOrigin[0]));
                imDispSize[0] = Math.round(imZoom*imActSize[0]);
                imOrigin[1] = Math.round(y-factor*(y-imOrigin[1]));
                imDispSize[1] = Math.round(imZoom*imActSize[1]);
                redraw();
            }
        }
    }
}

function mouseMove(event) {
    // socket.emit('client2ServerMessage', 'mouse move');
    if (mouseIsDown) {
        var x = event.layerX;
        var y = event.layerY;
        if (movingPlaneSlider) {
            movePlaneSlider(y);
        } else if (movingChannelSlider) {
            moveChannelSlider(y);
        } else {
            if (currentStateIcon == iconPointer) { // annotating
                annotateAtPoint(x,y);
            } else if (currentStateIcon == iconMove) {
                if (x >= imOrigin[0] && x <= imOrigin[0]+imDispSize[0] && y >= imOrigin[1] && y <= imOrigin[1]+imDispSize[1]) {
                    imOrigin[0] = imOriginBeforeMoving[0]+x-mouseDownLoc[0];
                    imOrigin[1] = imOriginBeforeMoving[1]+y-mouseDownLoc[1];
                    redraw();
                }
            }
        }
    }
}

function annotateAtPoint(x,y) {
    var j = Math.floor((x-imOrigin[0])/imZoom);
    var i = Math.floor((y-imOrigin[1])/imZoom);
    if (i >= pencilSize && i < imActSize[1]-pencilSize && j >= pencilSize && j < imActSize[0]-pencilSize) {
        for (var ii = i-pencilSize; ii <= i+pencilSize; ii++) {
            for (var jj = j-pencilSize; jj <= j+pencilSize; jj++) {
                if (Math.sqrt(Math.pow(jj-j,2)+Math.pow(ii-i,2)) < pencilSize) {
                    var index = ii*maskCanvas.width*4+jj*4+3;
                    var lmIndex = ii*maskCanvas.width+jj;
                    if (pencilDraws) {
                        maskData[index] = 127;
                        labelMask[lmIndex] = annotationClass;
                        if (labelMaskForInterp) { labelMaskForInterp[lmIndex] = annotationClass; }
                    } else {
                        maskData[index] = 0;
                        if (labelMask[lmIndex] == annotationClass) {
                            labelMask[lmIndex] = 0;
                            if (labelMaskForInterp) { labelMaskForInterp[lmIndex] = 0; }
                        }
                    }
                }
            }
        }
        maskContext.putImageData(maskImageData,0,0,j-pencilSize,i-pencilSize,2*pencilSize,2*pencilSize);
        // only painting 'dirty' rect
        // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/putImageData
        redrawImageROI(j-pencilSize,i-pencilSize,2*pencilSize,2*pencilSize);
    }
}

function resizeHTMLCanvas() {
    htmlCanvas.width = window.innerWidth-canvasWidthDiffToWindow;
    htmlCanvas.height = window.innerHeight-canvasHeightDiffToWindow;
    iconsArea.width = window.innerWidth-canvasWidthDiffToWindow;
    dialogArea.style.height = window.innerHeight-canvasHeightDiffToWindow;
    docButton.style.left = window.innerWidth-canvasWidthDiffToWindow-52;
    canvasCenter = [htmlCanvas.width/2, htmlCanvas.height/2];
    centeredImageCoordinates();
    redraw();

    svgDiv.style.width = htmlCanvas.width;
    svgDiv.style.height = htmlCanvas.height;
    svg.setAttribute('width',htmlCanvas.width);
    svg.setAttribute('height',htmlCanvas.height);

    if (planeSliderPresent) { removePlaneSlider(); drawPlaneSlider(); }
    if (channelSliderPresent) { removeChannelSlider(); drawChannelSlider(); };
}

function redraw() {
    context.clearRect(0, 0, htmlCanvas.width, htmlCanvas.height);
    // context.imageSmoothingEnabled = false;
    if (image) {
        context.drawImage(image, imOrigin[0], imOrigin[1], imDispSize[0], imDispSize[1]);
    } else if (imageCanvas) {
        context.drawImage(imageCanvas, imOrigin[0], imOrigin[1], imDispSize[0], imDispSize[1]);
    }
    
    if (maskCanvas) {
        context.globalCompositeOperation = 'source-over';
        context.drawImage(maskCanvas,imOrigin[0],imOrigin[1],imDispSize[0],imDispSize[1]);
    }
}

function redrawImageROI(x0,y0,w,h) {
    x1 = Math.round(imOrigin[0]+x0*imZoom);
    y1 = Math.round(imOrigin[1]+y0*imZoom);
    ww = Math.round(w*imZoom);
    hh = Math.round(h*imZoom);
    context.clearRect(x1,y1,ww,hh);
    if (image) {
        context.drawImage(image,x0,y0,w,h,x1,y1,ww,hh);
    } else if (imageCanvas) {
        context.drawImage(imageCanvas,x0,y0,w,h,x1,y1,ww,hh);    
    }
    if (maskCanvas) {
        context.drawImage(maskCanvas,x0,y0,w,h,x1,y1,ww,hh);
    }
    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
}

// ----------------------------------------------------------------------------------------------------
// tools
// ----------------------------------------------------------------------------------------------------

// appends 'Cancel' button where dismissing the tool doesn't need to perform other tasks (e.g. clearing masks)
function appendDismissButton(topOffset) {
    var dismiss = document.createElement('INPUT');
    dismiss.setAttribute('type','button');
    $(dismiss).button();
    // dismiss.addEventListener('click', hideTool, false);
    dismiss.onclick = function () { hideTool(); }
    dismiss.style.top = topOffset;
    dismiss.value = 'Cancel';
    dismiss.style.color = 'rgb(127,0,0)';
    // dismiss.style.backgroundColor = 'rgb(220,220,220)';
    tool.appendChild(dismiss);
}

async function showTool() {
    tool = document.getElementById('t');

    var i0 = -parseInt(tool.style.height);
    var i1 = 10;

    tool.style.opacity = '1';
    for (var i = i0; i <= i1; i += 10) {
        await sleep(10);
        tool.style.bottom = i;
        var factor = (i-i0)/(i1-i0);
        dialogArea.style.opacity = (1-factor)+factor*0.1;
    }

    textInput.disabled = true;
}

async function hideTool(tool) {
    tool = document.getElementById('t');

    var i0 = 10;
    var i1 = -parseInt(tool.style.height);
    for (var i = i0; i >= i1; i -= 10) {
        await sleep(10);
        tool.style.bottom = i;
        var factor = (i0-i)/(i0-i1);
        dialogArea.style.opacity = (1-factor)*0.1+factor;
    }
    tool.style.opacity = '0';
    disableIconPointer();

    textInput.disabled = false;
    textInput.focus();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ----------------------------------------------------------------------------------------------------
// tool: find spots

function toolFindSpots() {
    tool = document.getElementById('t');
    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    labelStrings = ['Sigma','Threshold'];
    var sliderDefaults = [2,0.5];
    sliderRangesMin = [1,0]
    sliderRangesMax = [10,1]
    for (var i = 0; i < labelStrings.length; i++) {
        var toolLabel = document.createElement('LABEL');
        toolLabel.innerHTML = labelStrings[i]+': '+sliderDefaults[i].toFixed(1);
        toolLabel.style.left = 10;
        toolLabel.style.top = 10+i*45;
        tool.appendChild(toolLabel);
        var toolSlider = document.createElement('INPUT');
        toolSlider.setAttribute('type','range');
        toolSlider.setAttribute('min','0');
        toolSlider.setAttribute('max','100');
        var sliderPosition = Math.round((sliderDefaults[i]-sliderRangesMin[i])/(sliderRangesMax[i]-sliderRangesMin[i])*100);
        toolSlider.setAttribute('value',sliderPosition.toString());
        toolSlider.style.top = 20+i*45;
        tool.appendChild(toolSlider);
    }
    tool.childNodes[1].oninput = function () { toolSliderChangeParameter(this.value, 0 ,labelStrings[0], 1, 10); }
    tool.childNodes[3].oninput = function () { toolSliderChangeParameter(this.value, 2 ,labelStrings[1], 0, 1); }

    var buttonWidth = (tool.clientWidth-30)/2

    var buttonCompute = document.createElement('INPUT');
    buttonCompute.setAttribute('type','button');
    $(buttonCompute).button();
    buttonCompute.style.top = 20+labelStrings.length*45;
    buttonCompute.style.width = buttonWidth;
    buttonCompute.value = 'Compute';
    buttonCompute.onclick = function () { findSpotsButtonClick('compute'); }
    tool.appendChild(buttonCompute);

    var buttonDone = document.createElement('INPUT');
    buttonDone.setAttribute('type','button');
    $(buttonDone).button();
    buttonDone.style.top = 20+labelStrings.length*45;
    buttonDone.style.width = buttonWidth;
    buttonDone.style.left = buttonWidth+20;
    buttonDone.value = 'Done';
    buttonDone.onclick = function () { findSpotsButtonClick('done'); }
    tool.appendChild(buttonDone);

    var buttonCancel = document.createElement('INPUT');
    buttonCancel.setAttribute('type','button');
    $(buttonCancel).button();
    buttonCancel.style.top = 60+labelStrings.length*45;
    buttonCancel.value = 'Cancel';
    buttonCancel.onclick = function () { findSpotsButtonClick('cancel'); }
    buttonCancel.style.color = 'rgb(127,0,0)';
    tool.appendChild(buttonCancel);

    tool.style.height = 100+labelStrings.length*45;

    if (!maskCanvas) { setMaskCanvas(); } // we'll draw spot locations on maskCanvas
    showTool();
}

function toolSliderChangeParameter(value,childNodesIndex,sliderLabel,rangeMin,rangeMax) {
    var num = rangeMin+value/100*(rangeMax-rangeMin)
    tool.childNodes[childNodesIndex].innerHTML = sliderLabel+': '+num.toFixed(1);
}

function findSpotsButtonClick(buttonID) {
    if (buttonID == 'compute') {
        var sliderValues = []; for (var i = 1; i <= 3; i += 2 ) { sliderValues.push(parseInt(tool.childNodes[i].value)); }

        var prmt0 = sliderRangesMin[0]+sliderValues[0]/100*(sliderRangesMax[0]-sliderRangesMin[0]); // sigma
        var prmt1 = sliderRangesMin[1]+sliderValues[1]/100*(sliderRangesMax[1]-sliderRangesMin[1]); // threshold

        var spotdetprmts = [prmt0, prmt1];
        // upload to server
        var formdata = new FormData();
        formdata.append('text', JSON.stringify({description: 'spotdetprmts', content: spotdetprmts}));
        $.ajax({
            url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
            success: function(data) // data = [[r0,c0],[r1,c1],...,[rN,cN]]
            {
                drawSpots(data);
            }
        });
    } else if (buttonID == 'done') {
        socket.emit('client2ServerMessage', 'toolFindSpotsButton done');
        maskCanvas = null; // mark for garbage collection (to save memory)
        redraw(); hideTool();
    } else if (buttonID == 'cancel') {
        socket.emit('client2ServerMessage', 'toolFindSpotsButton cancel');
        maskCanvas = null; // mark for garbage collection (to save memory)
        redraw(); hideTool();
    }
}

function drawSpots(data) {
    clearMask();
    var sigma = Math.round(sliderRangesMin[0]+parseInt(tool.childNodes[1].value)/100*(sliderRangesMax[0]-sliderRangesMin[0]));
    for (var i = 0; i < data.length; i ++) {
        var row0 = data[i][0];
        var col0 = data[i][1];
        for (var j = 0; j < Math.ceil(2*Math.PI*sigma); j++) {
            var row = Math.round(row0+sigma*Math.cos(j));
            var col = Math.round(col0+sigma*Math.sin(j));
            var index = row*maskCanvas.width*4+col*4+3;
            maskData[index] = 255;
        }
        maskContext.putImageData(maskImageData,0,0,col0-sigma,row0-sigma,2*sigma+1,2*sigma+1);
        redrawImageROI(col0-sigma,row0-sigma,2*sigma+1,2*sigma+1);
    }
    // maskContext.putImageData(maskImageData,0,0);
    // redraw();
}

// ----------------------------------------------------------------------------------------------------
// tool: segment (via basic threshold)

function toolThrSegment() {
    tool = document.getElementById('t');
    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    var pmIndexLabel = document.createElement('LABEL');
    pmIndexLabel.innerHTML = 'Class of Pixels to Segment';
    pmIndexLabel.style.left = 10;
    pmIndexLabel.style.top = 10;
    tool.appendChild(pmIndexLabel);

    var selectmenu = document.createElement('SELECT');
    selectmenu.setAttribute('name','selectmenu');
    selectmenu.setAttribute('id','selectmenu');
    var classLabels = ['Dark Pixels','Bright Pixels'];
    for (var i = 0; i < 2; i++) {
        var option = document.createElement('OPTION');
        option.value = i+1;
        option.innerHTML = classLabels[i];
        selectmenu.appendChild(option);
    }
    selectmenu.style.top = 30;
    tool.appendChild(selectmenu);

    // var labelStrings = ['Smoothness','Threshold','Fragmentation'];
    labelStrings = ['Smoothness','Threshold'];
    // var defaultSliderPositions = [50,33,67];
    var defaultSliderPositions = [0,50];
    for (var i = 0; i < labelStrings.length; i++) {
        var toolLabel = document.createElement('LABEL');
        toolLabel.innerHTML = labelStrings[i];
        toolLabel.style.left = 10;
        toolLabel.style.top = 70+i*45;
        tool.appendChild(toolLabel);
        var toolSlider = document.createElement('INPUT');
        toolSlider.setAttribute('type','range');
        toolSlider.setAttribute('min','0');
        toolSlider.setAttribute('max','100');
        toolSlider.setAttribute('value',defaultSliderPositions[i].toString());
        toolSlider.style.top = 80+i*45;
        tool.appendChild(toolSlider);
    }

    var buttonWidth = (tool.clientWidth-30)/2

    var buttonCompute = document.createElement('INPUT');
    buttonCompute.setAttribute('type','button');
    $(buttonCompute).button();
    buttonCompute.style.top = 80+labelStrings.length*45;
    buttonCompute.style.width = buttonWidth;
    buttonCompute.value = 'Compute';
    buttonCompute.onclick = function () { thrSegmentButtonClick('compute'); }
    tool.appendChild(buttonCompute);

    var buttonDone = document.createElement('INPUT');
    buttonDone.setAttribute('type','button');
    $(buttonDone).button();
    buttonDone.style.top = 80+labelStrings.length*45;
    buttonDone.style.width = buttonWidth;
    buttonDone.style.left = buttonWidth+20;
    buttonDone.value = 'Done';
    buttonDone.onclick = function () { thrSegmentButtonClick('done'); }
    tool.appendChild(buttonDone);

    var buttonCancel = document.createElement('INPUT');
    buttonCancel.setAttribute('type','button');
    $(buttonCancel).button();
    buttonCancel.style.top = 120+labelStrings.length*45;
    buttonCancel.value = 'Cancel';
    buttonCancel.onclick = function () { thrSegmentButtonClick('cancel'); }
    buttonCancel.style.color = 'rgb(127,0,0)';
    tool.appendChild(buttonCancel);

    tool.style.height = 160+labelStrings.length*45;

    showTool();
}

function thrSegmentButtonClick(buttonID) {
    if (buttonID == 'compute') {
        socket.emit('client2ServerMessage', 'set mask type segmMask');
        var thrsegprmts = []; for (var i = 1; i <= 5; i += 2 ) { thrsegprmts.push(parseInt(tool.childNodes[i].value)); }
        // upload to server
        var formdata = new FormData();
        formdata.append('text', JSON.stringify({description: 'thrsegprmts', content: thrsegprmts}));
        $.ajax({
            url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
            success: function(data)
            {
                setPlane(data.tcpi,data.shape,data.data,data.planeView,data.mask,data.maskType,false);
            }
        });
    } else if (buttonID == 'done') {
        socket.emit('client2ServerMessage', 'set mask type noMask'); labelMask = null; clearMask(); hideTool(); segmentStep = 0;
        var thrsegprmts = ''; for (var i = 1; i <= 5; i += 2 ) { thrsegprmts += ' '+tool.childNodes[i].value; }
        socket.emit('client2ServerMessage', 'threshold'+thrsegprmts);
    } else if (buttonID == 'cancel') {
        socket.emit('client2ServerMessage', 'set mask type noMask'); labelMask = null; clearMask(); hideTool(); segmentStep = 0;
    }
}

// ----------------------------------------------------------------------------------------------------
// tool: segment (via ml segmenter)

function toolMLSegment() {
    tool = document.getElementById('t');
    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    var pmIndexLabel = document.createElement('LABEL');
    pmIndexLabel.innerHTML = 'Object Class Index';
    pmIndexLabel.style.left = 10;
    pmIndexLabel.style.top = 10;
    tool.appendChild(pmIndexLabel);

    var selectmenu = document.createElement('SELECT');
    selectmenu.setAttribute('name','selectmenu');
    selectmenu.setAttribute('id','selectmenu');
    for (var i = 0; i < nClasses; i++) {
        var option = document.createElement('OPTION');
        option.value = i+1;
        option.innerHTML = 'Class '+(i+1);
        selectmenu.appendChild(option);
    }
    selectmenu.style.top = 30;
    tool.appendChild(selectmenu);

    var labelStrings = ['Smoothness','Threshold'];
    var defaultSliderPositions = [0,33];
    if (nPlanes > 0) { defaultSliderPositions[0] = 0; }
    for (var i = 0; i < labelStrings.length; i++) {
        var toolLabel = document.createElement('LABEL');
        toolLabel.innerHTML = labelStrings[i];
        toolLabel.style.left = 10;
        toolLabel.style.top = 70+i*45;
        tool.appendChild(toolLabel);
        var toolSlider = document.createElement('INPUT');
        toolSlider.setAttribute('type','range');
        toolSlider.setAttribute('min','0');
        toolSlider.setAttribute('max','100');
        toolSlider.setAttribute('value',defaultSliderPositions[i].toString());
        toolSlider.style.top = 80+i*45;
        tool.appendChild(toolSlider);
    }

    var buttonWidth = (tool.clientWidth-30)/2

    var buttonCompute = document.createElement('INPUT');
    buttonCompute.setAttribute('type','button');
    $(buttonCompute).button();
    buttonCompute.style.top = 80+labelStrings.length*45;
    buttonCompute.style.width = buttonWidth;
    buttonCompute.value = 'Compute';
    buttonCompute.onclick = function () { mlSegmentButtonClick('compute'); }
    tool.appendChild(buttonCompute);

    var buttonDone = document.createElement('INPUT');
    buttonDone.setAttribute('type','button');
    $(buttonDone).button();
    buttonDone.style.top = 80+labelStrings.length*45;
    buttonDone.style.width = buttonWidth;
    buttonDone.style.left = buttonWidth+20;
    buttonDone.value = 'Done';
    buttonDone.onclick = function () { mlSegmentButtonClick('done'); }
    tool.appendChild(buttonDone);

    var buttonCancel = document.createElement('INPUT');
    buttonCancel.setAttribute('type','button');
    $(buttonCancel).button();
    buttonCancel.style.top = 120+labelStrings.length*45;
    buttonCancel.value = 'Cancel';
    buttonCancel.onclick = function () { mlSegmentButtonClick('cancel'); }
    buttonCancel.style.color = 'rgb(127,0,0)';
    tool.appendChild(buttonCancel);

    tool.style.height = 160+labelStrings.length*45;

    showTool();
}

function mlSegmentButtonClick(buttonID) {
    if (buttonID == 'compute') {
        socket.emit('client2ServerMessage', 'set mask type segmMask');
        var mlsegprmts = [];
        for (var i = 1; i <= 5; i += 2 ) { mlsegprmts.push(parseInt(tool.childNodes[i].value)); }
        // upload to server
        var formdata = new FormData();
        formdata.append('text', JSON.stringify({description: 'mlsegprmts', content: mlsegprmts}));
        $.ajax({
            url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
            success: function(data)
            {
                setPlane(data.tcpi,data.shape,data.data,data.planeView,data.mask,data.maskType,false);
            }
        });
    } else if (buttonID == 'done') {
        socket.emit('client2ServerMessage', 'set mask type noMask'); labelMask = null; clearMask(); hideTool(); segmentStep = 0;
        var mlsegprmts = ''; for (var i = 1; i <= 5; i += 2 ) { mlsegprmts += ' '+tool.childNodes[i].value; }
        socket.emit('client2ServerMessage', 'mlthreshold'+mlsegprmts);
    } else if (buttonID == 'cancel') {
        socket.emit('client2ServerMessage', 'set mask type noMask'); labelMask = null; clearMask(); hideTool(); segmentStep = 0;
    }
}

// ----------------------------------------------------------------------------------------------------
// tool: train pixel/voxel classifier

function toolTrainMLSegmenter() {
    tool = document.getElementById('t');
    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    var edgeFeaturesLabel = document.createElement('LABEL');
    edgeFeaturesLabel.innerHTML = 'Edge/Ridge Features | Scales:';
    edgeFeaturesLabel.style.left = 30;
    edgeFeaturesLabel.style.top = 10;
    edgeFeaturesLabel.style.left = 10;
    tool.appendChild(edgeFeaturesLabel);

    var scales = [1,2,4,8];
    var vOffset = 30;
    var hStep = (tool.clientWidth-20)/scales.length;
    for (var i = 0; i < scales.length; i++) {
        var checkbox = document.createElement('INPUT');
        checkbox.setAttribute('type','checkbox');
        checkbox.setAttribute('id','edgeFeatCheckbox'+scales[i]);
        var checkboxLabel = document.createElement('LABEL');
        checkboxLabel.innerHTML = ''+scales[i];
        checkbox.style.left = 10+i*hStep;
        checkbox.style.top = vOffset;
        checkboxLabel.style.left = 30+i*hStep;
        checkboxLabel.style.top = vOffset+2;
        checkbox.checked = false;
        if (nPlanes == 0) {
            checkbox.checked = true;
        } else {
            if (i == 1) {
                checkbox.checked = true;
            }
        }
        tool.appendChild(checkbox);
        tool.appendChild(checkboxLabel);
    }

    var spotFeaturesLabel = document.createElement('LABEL');
    spotFeaturesLabel.innerHTML = 'Spot/Blob Features | Scales:';
    spotFeaturesLabel.style.left = 30;
    spotFeaturesLabel.style.top = 70;
    spotFeaturesLabel.style.left = 10;
    tool.appendChild(spotFeaturesLabel);

    scales = [2,4,8];
    vOffset = 90;
    hStep = (tool.clientWidth-20)/scales.length;
    for (var i = 0; i < scales.length; i++) {
        var checkbox = document.createElement('INPUT');
        checkbox.setAttribute('type','checkbox');
        checkbox.setAttribute('id','spotFeatCheckbox'+scales[i]);
        var checkboxLabel = document.createElement('LABEL');
        checkboxLabel.innerHTML = ''+scales[i];
        checkbox.style.left = 10+i*hStep;
        checkbox.style.top = vOffset;
        checkboxLabel.style.left = 30+i*hStep;
        checkboxLabel.style.top = vOffset+2;
        checkbox.checked = false;
        if (nPlanes == 0) {
            checkbox.checked = true;
        }
        tool.appendChild(checkbox);
        tool.appendChild(checkboxLabel);
    }

    var trainButton = document.createElement('INPUT');
    trainButton.setAttribute('type','button');
    $(trainButton).button();
    trainButton.addEventListener('click', trainButtonClicked, false);
    trainButton.style.top = 130;
    trainButton.value = 'Train';
    tool.appendChild(trainButton);

    appendDismissButton(180);

    tool.style.height = 220;
    showTool();
}

function trainButtonClicked() {
    var nodes = tool.childNodes;
    var edgeFeatPrmts = [];
    var spotFeatPrmts = [];
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].type == 'checkbox' && nodes[i].checked) {
            var id = nodes[i].id;
            if (id.substring(0,4) == 'edge') {
                var prm = parseInt(id.substring(16,id.length));
                edgeFeatPrmts.push(prm);
            } else if (id.substring(0,4) == 'spot') {
                var prm = parseInt(id.substring(16,id.length));
                spotFeatPrmts.push(prm);
            }
        }
    }

    var formdata = new FormData();
    formdata.append('text', JSON.stringify({description: 'mlsegtrainprmts', content: [edgeFeatPrmts, spotFeatPrmts]}));
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
        success: function(data)
        {
            addMessageToDialog(data.message, 'right');
            socket.emit('client2ServerMessage', 'train ml segmenter');
            hideTool();
        }
    });
}

// ----------------------------------------------------------------------------------------------------
// tool: annotate

function toolAnnotate(createLabelMaskOption) {
    // createLabelMaskOption = '0': create label mask on server (used when annotating from scratch)
    // createLabelMaskOption = '1': assumes label mask already created on server (used when editing annotations)

    pencilSize = 10; // should be integer > 0
    pencilDraws = true; // otherwise erases

    tool = document.getElementById('t');
    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    var draw = document.createElement('INPUT');
    draw.setAttribute('type','radio');
    draw.setAttribute('id','draw');
    var drawLabel = document.createElement('LABEL');
    drawLabel.innerHTML = 'Draw';
    drawLabel.setAttribute('for','draw');
    draw.style.left = 8;
    draw.style.top = 10;
    drawLabel.style.left = 30;
    drawLabel.style.top = 12;
    draw.checked = true;
    draw.onclick = function () { erase.checked = false; pencilDraws = true; }
    tool.appendChild(draw);
    tool.appendChild(drawLabel);

    var erase = document.createElement('INPUT');
    erase.setAttribute('type','radio');
    erase.setAttribute('id','erase');
    var eraseLabel = document.createElement('LABEL');
    eraseLabel.innerHTML = 'Erase';
    eraseLabel.setAttribute('for','erase');
    erase.style.left = 68;
    erase.style.top = 10;
    eraseLabel.style.left = 90;
    eraseLabel.style.top = 12;
    erase.onclick = function () { draw.checked = false; pencilDraws = false; }
    tool.appendChild(erase);
    tool.appendChild(eraseLabel);

    var slider = document.createElement('INPUT');
    slider.setAttribute('type','range');
    slider.setAttribute('name','slider');
    slider.setAttribute('id','slider');
    slider.setAttribute('min','1');
    slider.setAttribute('max','100');
    slider.setAttribute('value',pencilSize.toString());
    // $('#slider').slider();
    // $(slider).slider();
    slider.style.top = 35;
    slider.oninput = function () { changePencilSize(this.value); }
    slider.onmousedown = function () { drawSVGCircle(); }
    slider.onmouseup = function () { removeSVGCircle(); }
    slider.ontouchstart = function () { drawSVGCircle(); }
    slider.ontouchend = function () { removeSVGCircle(); }
    tool.appendChild(slider);

    var sliderValue = document.createElement('DIV');
    sliderValue.setAttribute('class','label');
    sliderValue.innerHTML = 'Pencil size: '+pencilSize.toString();
    sliderValue.style.top = 60;
    sliderValue.style.left = 15;
    tool.appendChild(sliderValue);

    var selectmenu = document.createElement('SELECT');
    selectmenu.setAttribute('name','selectmenu');
    selectmenu.setAttribute('id','selectmenu');
    for (var i = 0; i < nClasses; i++) {
        var option = document.createElement('OPTION');
        option.value = i+1;
        option.innerHTML = 'Class '+(i+1);
        selectmenu.appendChild(option);
    }
    selectmenu.style.top = 95;
    selectmenu.addEventListener('change', changeAnnotationClass, false);
    tool.appendChild(selectmenu);

    var vertOffset = 0;
    if (nClasses == 2 && !annotateWithInterpolation) { // button to uniformly sample complement of the other class
        var unifSampComp = document.createElement('INPUT');
        unifSampComp.setAttribute('type','button');
        $(unifSampComp).button();
        unifSampComp.addEventListener('click', uniformlySampleComplement, false);
        unifSampComp.style.top = 135;
        unifSampComp.value = 'Uniformly Sample Complement';
        vertOffset = 40;
        tool.appendChild(unifSampComp);
    }
    if (annotateWithInterpolation && nPlanes > 0) {
        var antWithInterp = document.createElement('INPUT');
        antWithInterp.setAttribute('type','button');
        $(antWithInterp).button();
        antWithInterp.addEventListener('click', interpolateAnnotations, false);
        antWithInterp.style.top = 135;
        antWithInterp.value = 'Interpolate Annotations';
        vertOffset = 40;
        tool.appendChild(antWithInterp);
    }

    var save = document.createElement('INPUT');
    save.setAttribute('type','button');
    $(save).button();
    save.addEventListener('click', saveMask, false);
    save.style.top = 150+vertOffset;
    save.value = 'Save Masks';
    tool.appendChild(save);

    var cancel = document.createElement('INPUT');
    cancel.setAttribute('type','button');
    $(cancel).button();
    cancel.style.top = 190+vertOffset;
    cancel.value = 'Cancel';
    cancel.onclick = function () {
        clearMask(); hideTool(); setImDisplayThresholds(0,255); annotateStep = 0;
        socket.emit('client2ServerMessage', 'set mask type noMask');
    }
    cancel.style.color = 'rgb(127,0,0)';
    tool.appendChild(cancel);
    
    tool.style.height = 230+vertOffset;

    labelMask = new Uint8ClampedArray(imActSize[0]*imActSize[1]); // only single plane is kept in memory
    if (annotateWithInterpolation) { labelMaskForInterp = new Uint8ClampedArray(imActSize[0]*imActSize[1]); }
    setMaskCanvas();
    annotationClass = 1;
    var options = ' '; options += createLabelMaskOption;
    if (!annotateWithInterpolation) { options += '0'; } else { options += '1' }
    socket.emit('client2ServerMessage', 'create label mask'+options); // tool will be shown when mask is allocated on server
}

function uniformlySampleComplement() {
    // only called when nClasses == 2
    // not called when annotateWithInterpolation = True
    var compClass = 1;
    if (annotationClass == 1) { compClass = 2; }
    nCompClassPixels = 0;
    for (var i = 0; i < imActSize[1]; i++) {
        for (var j = 0; j < imActSize[0]; j++) {
            var lmIndex = i*maskCanvas.width+j;
            if (labelMask[lmIndex] == compClass) {
                nCompClassPixels += 1;
            }
        }
    }
    var fraction = nCompClassPixels/(imActSize[0]*imActSize[1]);
    for (var i = 0; i < imActSize[1]; i++) {
        for (var j = 0; j < imActSize[0]; j++) {
            var index = i*maskCanvas.width*4+j*4+3;
            var lmIndex = i*maskCanvas.width+j;
            if (labelMask[lmIndex] == 0 && Math.random() < fraction) {
                maskData[index] = 127;
                labelMask[lmIndex] = annotationClass;
            }
        }
    }
    maskContext.putImageData(maskImageData,0,0);
    redraw();
}

function interpolateAnnotations() { // only called when nPlanes > 0
    uploadMaskPlane('interpolate');
}

function drawSVGCircle() {
    svgCircle = document.createElementNS(NS,'circle');
    svgCircle.setAttribute('cx',canvasCenter[0]);
    svgCircle.setAttribute('cy',canvasCenter[1]);
    svgCircle.setAttribute('r',0);
    svgCircle.style.stroke = 'white';
    svgCircle.setAttribute('fill-opacity',0);
    svg.appendChild(svgCircle);
    changePencilSize(pencilSize);
}

function removeSVGCircle() {
    svg.removeChild(svgCircle);
}

function changePencilSize(value) {
    tool.childNodes[5].innerHTML = 'Pencil size: '+value;
    pencilSize = parseInt(value);
    svgCircle.setAttribute('r',pencilSize*imZoom);
}

function changeAnnotationClass() {
    annotationClass = parseInt(tool.childNodes[6].value);
    updateMaskCanvas();
}

function updateMaskCanvas() {
    for (var i = 0; i < maskCanvas.height; i++) {
        for (var j = 0; j < maskCanvas.width; j++) {
            var index = i*maskCanvas.width*4+j*4+3;
            var lmIndex = i*maskCanvas.width+j;
            if (labelMask[lmIndex] == annotationClass) {
                maskData[index] = 127;
            } else {
                maskData[index] = 0;
            }
        }
    }
    maskContext.putImageData(maskImageData,0,0);
    redraw();
}

function saveMask() {
    annotateStep = 3;
    uploadMaskPlane('');
}

function uploadMaskPlane(nextStep) {
    console.log('next step:', nextStep)
    
    var ll = [];
    for (var i = 0; i < nClasses; i++) {
        l = [];
        ll.push(l);
    }
    for (var i = 0; i < labelMask.length; i++) {
        var lm = labelMask[i];
        if (lm > 0) {
            ll[lm-1].push(i);
        }
        labelMask[i] = 0;
    }

    var llI = [];
    if (labelMaskForInterp) {
        for (var i = 0; i < nClasses; i++) {
            l = [];
            llI.push(l);
        }
        for (var i = 0; i < labelMaskForInterp.length; i++) {
            var lm = labelMaskForInterp[i];
            if (lm > 0) {
                llI[lm-1].push(i);
            }
            labelMaskForInterp[i] = 0;
        }
    }

    // upload to server
    var formdata = new FormData();
    formdata.append('text', JSON.stringify({description: 'annotation', content: [ll, llI]}));
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonupload', type: 'POST', data: formdata, contentType: false, cache: false, processData:false,
        success: function(data)
        {
            if (annotateStep == 2) {
                if (nextStep == 'newPlane') {
                    console.log('now show new plane');
                    newPlaneFromServer(scheduledPlaneIndex[0],scheduledPlaneIndex[1],scheduledPlaneIndex[2]);
                } else if (nextStep == 'newView') {
                    console.log('now show new view');
                    socket.emit('client2ServerMessage', 'view plane '+scheduledPlaneView);
                } else if (nextStep == 'interpolate') {
                    hideTool();
                    addMessageToDialog('interpolating annotations...','right');
                    socket.emit('client2ServerMessage', 'interpolate plane masks');
                }
            } else if (annotateStep == 3) {
                console.log('done annotating')
                annotateStep = 0;
                if (nPlanes == 0) { //  server knows to save masks once annotations are uploaded
                    dismissAnnotationTool();
                } else { // tell server to save masks
                    socket.emit('client2ServerMessage', 'save label mask');
                }
            }
        }
    });
}

function dismissAnnotationTool() {
    addMessageToDialog('annotations uploaded', 'right');
    labelMask = null; // mark for garbage collection (to save memory)
    if (labelMaskForInterp) { labelMaskForInterp = null; }
    didAnnotateImages = true;
    clearMask(); hideTool(); setImDisplayThresholds(0,255);
}

function clearMask() {
    if (maskCanvas) {
        for(var i = 0; i < maskCanvas.height; i++) {
            for (var j = 0; j < maskCanvas.width; j++) {
                maskData[i*maskCanvas.width*4+j*4+3] = 0;
            }
        }
        maskContext.putImageData(maskImageData,0,0);    
    }
    if (labelMask) {
        for (var i = 0; i < labelMask.length; i++) { labelMask[i] = 0; }
    }
    redraw();
}

// ----------------------------------------------------------------------------------------------------
// tool: new image

function toolNewImage() {
    tool = document.getElementById('t');

    while (tool.firstChild) {
        tool.removeChild(tool.firstChild);
    }

    // var form  = document.createElement('FORM');
    // // form.setAttribute('method','POST');
    // form.method = 'POST';
    // form.enctype = 'multipart/form-data';

    // hidden
    var input = document.createElement('INPUT');
    input.setAttribute('type','file');
    input.style.opacity = '0';
    // https://developer.mozilla.org/en-US/docs/Web/API/File/Using_files_from_web_applications
    input.setAttribute('accept','.tif, .tiff, .TIF, .TIFF');
    input.addEventListener('change', handleFiles, false);
    // input.name = 'file';

    var browse = document.createElement('INPUT');
    browse.setAttribute('type','button');
    $(browse).button(); // applies jQuery styling
    // browse.setAttribute('onclick','input.click()');
    browse.addEventListener('click', fileBrowseClick, false);
    browse.style.top = '10';
    browse.value = 'Browse...';
    
    var label = document.createElement('DIV');
    label.setAttribute('class','label');
    label.innerHTML = 'No file selected.'
    label.style.top = '50';
    label.style.left = 13;

    var submit = document.createElement('INPUT');
    submit.setAttribute('type','button');
    $(submit).button();
    submit.addEventListener('click', fileSubmitClick, false);
    submit.style.top = '90';
    submit.value = 'Submit';

    // form.appendChild(input);
    tool.appendChild(input);
    tool.appendChild(browse);
    tool.appendChild(label);
    tool.appendChild(submit);

    appendDismissButton(140);

    tool.style.height = 180;
    
    showTool();
}

function fileBrowseClick() {
    tool.childNodes[0].click();
    // tool.childNodes[0].childNodes[0].click();
}

function handleFiles() {
    var file = this.files[0];
    if (file.name.length > 32) {
        tool.childNodes[2].innerHTML = file.name.substring(0,16) + ' ... ' + file.name.substring(file.name.length-16,file.name.length);
    } else {
        tool.childNodes[2].innerHTML = file.name;
    }
    // console.log('name: ' + file.name.substring(0,10), 'size: ' + file.size, 'type: ' + file.type);
}

function fileSubmitClick() {
    var formdata = new FormData();
    file = tool.childNodes[0].files[0];
    formdata.append('file', file);
    $.ajax({
        url: $SCRIPT_ROOT + '/jsonfileupload',  // Url to which the request is send
        type: 'POST',                           // Type of request to be send, called as method
        data: formdata,                         // Data sent to server, a set of key/value pairs (i.e. form fields and values)
        contentType: false,                     // The content type used when sending data to the server.
        cache: false,                           // To unable request pages to be cached
        processData:false,                      // To send DOMDocument or non processed data file it is set to false
        success: function(data)                 // A function to be called if request succeeds
        {
            setPlane(data.tcpi,data.shape,data.data,data.planeView,data.mask,data.maskType,true);
            socket.emit('dialog', 'image properties');
            hideTool();
        }
    });
}

function setPlane(tcpi,shape,data,currentViewPlane,mask,maskType,newImage) {
    // tcpi: time, channel, plane indices; newImage: if it's a new image (true) or just a new plane (false)
    didSetImage = true;
    var centImCoord = true;
    if (newImage) {
        if (planeSliderPresent) { removePlaneSlider(); }
        if (channelSliderPresent) { removeChannelSlider(); }
        thrButton.style.opacity = 0.5;
        thrButton.disabled = false;
        thrSlider.style.opacity = 0.5;
        thrSlider.disabled = false;
        thrButton.value = '.[...';
        thrLower = 0; thrUpper = 255;
        thrSlider.value = thrLower;
    }

    if (shape.length == 2) {
        nPlanes = nChannels = 0; // technically 1 but this means the image is a 2D tensor
        imActSize = [shape[1], shape[0]];
    } else if (shape.length == 3) {
        nPlanes = shape[0]; nChannels = 0;
        planeIndex = tcpi[2]; previousPlaneIndex = planeIndex; // used to avoid redundant refreshes
        channelIndex = 0; timeIndex = 0;
        imActSize = [shape[2], shape[1]];
        if (planeSliderPresent) {
            centImCoord = false;
        } else {
            planeView = currentViewPlane;
            drawPlaneSlider();
        }
    } else if (shape.length == 4) {
        nPlanes = shape[0]; nChannels = shape[1];
        planeIndex = tcpi[2]; previousPlaneIndex = planeIndex; // used to avoid redundant refreshes
        channelIndex = tcpi[1]; previousChannelIndex = channelIndex;
        timeIndex = 0;
        imActSize = [shape[3], shape[2]];
        if (planeSliderPresent) {
            centImCoord = false;
        } else {
            drawPlaneSlider();
        }
        if (!channelSliderPresent) {
            drawChannelSlider();
        }
    }

    if (newImage) {
        setImageCanvas();
        maskCanvas = labelMask = null;
    }

    contextImageData = imageContext.getImageData(0,0,imageCanvas.width,imageCanvas.height);
    imageData = contextImageData.data;
    for(var i = 0; i < imageCanvas.height; i++) {
        for (var j = 0; j < imageCanvas.width; j++) {
            var index  = i*imageCanvas.width*4+j*4;
            var d = data[i][j]; // assumed int in 0...255

            if (!newImage) {
                if (d < thrLower) { d = 0; }
                else if (d > thrUpper) { d = 255; }
                else { d = (d-thrLower)/(thrUpper-thrLower)*255; }
            }

            imageData[index]   = d;
            imageData[index+1] = d;
            imageData[index+2] = d;
            imageData[index+3] = 255;
            originalImageData[i*imageCanvas.width+j] = d;
        }
    }
    imageContext.putImageData(contextImageData,0,0);

    if (mask.length > 0) {
        // console.log('new image: ', newImage, 'mask size: ', mask.length,mask[0].length);
        if (maskType == 'labelMask') {
            setLabelMask(mask); updateMaskCanvas();
        } else if (maskType == 'segmMask') {
            setLabelMask(mask); annotationClass = 1; updateMaskCanvas();
        }
    } else {
        maskCanvas = labelMask = null;
    }

    image = null; // so that default image is not drawn
    if (centImCoord) { centeredImageCoordinates(); }
    redraw();
}

function setLabelMask(mask) {
    if (!maskCanvas) { setMaskCanvas(); }
    if (!labelMask) { labelMask = new Uint8ClampedArray(imActSize[0]*imActSize[1]); }
    for(var i = 0; i < maskCanvas.height; i++) {
        for (var j = 0; j < maskCanvas.width; j++) {
            labelMask[i*maskCanvas.width+j] = mask[i][j];
        }
    }
}

function setImageCanvas() {
    // either a canvas hasn't been set yet, or a new canvas size is needed
    if (!imageCanvas || !(imageCanvas.width == imActSize[0] && imageCanvas.height == imActSize[1])) {
        imageCanvas = document.createElement('canvas');
        imageCanvas.width = imActSize[0];
        imageCanvas.height = imActSize[1];
        imageContext = imageCanvas.getContext('2d');
        originalImageData = new Uint8ClampedArray(imActSize[0]*imActSize[1]);
        // needed because imageData (see below) is modified during display under various thresholds
    }
}

function setMaskCanvas() {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = imActSize[0];
    maskCanvas.height = imActSize[1];
    maskContext = maskCanvas.getContext('2d');

    maskImageData = maskContext.getImageData(0,0,maskCanvas.width,maskCanvas.height);
    maskData = maskImageData.data;
    for(var i = 0; i < maskCanvas.height; i++) {
        for (var j = 0; j < maskCanvas.width; j++) {
            var index  = i*maskCanvas.width*4+j*4;
            maskData[index]   = 0;
            maskData[index+1] = 0;
            maskData[index+2] = 255;
            maskData[index+3] = 0;
        }
    }
    maskContext.putImageData(maskImageData,0,0);
}

})();