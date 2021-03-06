import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

import 'Controllers.dart';

class TextToSpeech extends StatefulWidget {
  final MyStateController controller;
  Function() startHandler, stopHandler;

  TextToSpeech(
      {@required this.controller, this.startHandler, this.stopHandler});

  @override
  _TextToSpeechState createState() => _TextToSpeechState(controller);
}

class _TextToSpeechState extends State<TextToSpeech> {
  bool _isActive = false, _isBusy = false;

  final flutterTts = FlutterTts();
  MyStateController _controller;

  _TextToSpeechState(this._controller) {
    _controller.disable = () => setState(() {
          if (_isActive) stopSpeaking();
          _isActive = false;
          _isBusy = true;
        });
    _controller.enable = () => setState(() => _isBusy = false);
  }

  @override
  void initState() {
    flutterTts.setStartHandler(() {
      widget.startHandler();
      setState(() {
        _isActive = true;
      });
    });
    var stophandler = () {
      widget.stopHandler();
      setState(() {
        _isActive = false;
      });
    };
    flutterTts.setCompletionHandler(stophandler);
    flutterTts.setCancelHandler(stophandler);
    flutterTts.setErrorHandler((_) => stophandler());
  }

  @override
  Widget build(BuildContext context) {
    if (_isActive) {
      return RawMaterialButton(
        elevation: 2.0,
        fillColor: Colors.blueAccent,
        onPressed: _isBusy ? null : stopSpeaking,
        child: const Icon(
          Icons.speaker,
          color: Colors.white,
        ),
        shape: CircleBorder(),
        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: EdgeInsets.all(8),
      );
    } else {
      return IconButton(
        icon: const Icon(Icons.speaker),
        tooltip: 'speak for you',
        onPressed: _isBusy
            ? null
            : () async {
                await speak(_controller.Text);
              },
      );
    }
  }

  void speak(String text) async {
    var result = await flutterTts.speak(text);
  }

  void stopSpeaking() async {
    var result = await flutterTts.stop();
  }
}
