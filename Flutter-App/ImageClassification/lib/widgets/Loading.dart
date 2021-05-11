import 'package:flutter/material.dart';
import 'package:loading/indicator/ball_pulse_indicator.dart';
import 'package:loading/loading.dart';

class LoadingOfResponse extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white10,
      child: Center(
        child: Loading(
            indicator: BallPulseIndicator(), size: 100.0, color: Colors.blue),
      ),
    );
  }
}
