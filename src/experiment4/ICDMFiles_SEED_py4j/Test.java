import java.util.Random;

class Test
{
    public static void main(String args[]) {
        int seed = 0;
        Random rnd = new Random(seed);
        System.out.println(rnd.nextDouble());
    }
}
