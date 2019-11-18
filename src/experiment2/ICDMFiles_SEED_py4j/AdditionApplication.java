import py4j.GatewayServer;

public class AdditionApplication{
  public class Hoge {
    public Hoge() {

    }    
  }

  public int addition(int first, int second) {
    return first + second;
  }

  public static void main(String[] args) {
    AdditionApplication app = new AdditionApplication();
    GatewayServer server = new GatewayServer(app);
    //server.start();
    //System.out.println("Gateway Server Started");
  }
}
