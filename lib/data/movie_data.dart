import 'dart:convert';

import 'package:movie_website/model/movie_model.dart';
import 'package:http/http.dart' as http;

class MovieData {
  final String baseUrl = 'https://api.themoviedb.org/3/movie';
  final String bearerToken =
      'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ZDZmNWFlMzEzZTQ2NDZiODIxZDdjZGQzNmY4NmY0NCIsIm5iZiI6MTcyOTQzMTE0Ny41OTg5MjUsInN1YiI6IjY3MTUwMTAzMmJiYmE2NWY3YjExNDMyNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.PlkJv9gO3EJYkyF_gJK6MtB76FABCDCeVjbFutOpsF0';

  Future<List<MovieModel>> fetchTopRatedMovie() async {
    final response = await http.get(
      Uri.parse('$baseUrl/top_rated?language=en-US&page=1'),
      headers: {
        'Authorization': 'Bearer $bearerToken',
        'accept': 'application/json'
      },
    );
    if (response.statusCode == 200) {
      return ((jsonDecode(response.body)['results']) as List)
          .map((e) => MovieModel.fromJson(e))
          .toList();
    } else {
      throw Exception('Failed to load movie data');
    }
  }
}
