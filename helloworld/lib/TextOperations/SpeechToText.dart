import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'Controllers.dart';
import 'package:speech_recognition/speech_recognition.dart';

class SpeechToText extends StatefulWidget {
  Function(String) callback;
  Function() startHandler, stopHandler;
  MyStateController controller;

  SpeechToText(
      {this.callback,
      this.startHandler,
      this.stopHandler,
      @required this.controller});

  @override
  _SpeechToTextState createState() => _SpeechToTextState();
}

class _SpeechToTextState extends State<SpeechToText> {
  SpeechRecognition _speech;
  bool _isActive = false, _isBusy = false;

  void initState() {
    widget.controller.enable = () => setState(() => _isBusy = false);
    widget.controller.disable = () => setState(() => _isBusy = true);
    _speech = SpeechRecognition()
      ..setAvailabilityHandler((bool result) {})
      ..setCurrentLocaleHandler((String locale) {})
      ..setRecognitionStartedHandler(() {
        widget.startHandler();
        setState(() => _isActive = true);
      })
      ..setRecognitionResultHandler((String text) {
        print("---------->$text");
        widget.callback(text);
      })
      ..setRecognitionCompleteHandler(() {
        widget.stopHandler();
        setState(() => _isActive = false);
      })
      ..activate();
  }

  @override
  Widget build(BuildContext context) {
    if (_isActive) {
      return RawMaterialButton(
        elevation: 2.0,
        fillColor: Colors.blueAccent,
        onPressed: _isBusy ? null : _listen,
        child: const Icon(
          Icons.mic,
          color: Colors.white,
        ),
        shape: CircleBorder(),
        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
        padding: EdgeInsets.all(8),
      );
    } else {
      return IconButton(
        icon: const Icon(Icons.mic),
        tooltip: 'listen for you',
        onPressed: _isBusy ? null : _listen,
      );
    }
  }

  void _listen() async {
    if (!_isActive) {
      _speech.listen(locale: 'en_IN');
    } else {
      _speech.stop();
    }
  }
}
