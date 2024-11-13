import 'package:flutter/material.dart';
import 'package:gemini_chat/onboarding.dart';
import 'package:gemini_chat/themes.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: lightMode,
      darkTheme: darkMode,
      home: const Onboarding(),
    );
  }
}
