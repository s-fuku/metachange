import java.util.Random;

class TestRnd {
  public TestRnd () {
  }

  public static void main(String args[]) {
    Random rnd = new Random(1);
    int count = 0;
    int N = 1000000;
    for (int i = 0; i < N; ++i) {
      if (rnd.nextDouble() < 0.8) {
        count++;
      }
    }
    System.out.println((double)count/N);
  }
}
