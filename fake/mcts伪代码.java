import java.util.*;

class Node {
    public BoardState state;
    public Node parent;
    public Map<Action, Node> children = new HashMap<>();
    public double P; // 神经网络先验概率
    public int N;    // 动作访问次数
    public double W; // 累积胜利值
    public double Q; // 平均胜利值

    public Node(BoardState state, Node parent, double prior) {
        this.state = state;
        this.parent = parent;
        this.P = prior;
        this.N = 0;
        this.W = 0;
        this.Q = 0;
    }
}

public class MCTS {

    public static Action selectAction(Node node, double c_puct) {
        double bestScore = -Double.MAX_VALUE;
        Action bestAction = null;
        for (Map.Entry<Action, Node> entry : node.children.entrySet()) {
            Node child = entry.getValue();
            double score = child.Q + c_puct * child.P * Math.sqrt(node.N) / (1 + child.N);
            if (score > bestScore) {
                bestScore = score;
                bestAction = entry.getKey();
            }
        }
        return bestAction;
    }

    public static void mctsSearch(Node root, NeuralNetwork nn, int numSimulations, double c_puct) {
        for (int i = 0; i < numSimulations; i++) {
            Node node = root;
            List<Node> path = new ArrayList<>();
            path.add(node);

            // 1. 选择
            while (!node.children.isEmpty()) {
                Action action = selectAction(node, c_puct);
                node = node.children.get(action);
                path.add(node);
            }

            // 2. 扩展
            double value;
            if (!node.state.isTerminal()) {
                Map<Action, Double> policy = nn.predictPolicy(node.state);
                value = nn.predictValue(node.state);
                for (Map.Entry<Action, Double> p : policy.entrySet()) {
                    if (!node.children.containsKey(p.getKey())) {
                        node.children.put(p.getKey(), new Node(node.state.nextState(p.getKey()), node, p.getValue()));
                    }
                }
            } else {
                value = node.state.getReward();
            }

            // 3. 回溯
            for (int j = path.size() - 1; j >= 0; j--) {
                Node n = path.get(j);
                n.N += 1;
                n.W += value;
                n.Q = n.W / n.N;
                value = -value; // 交替玩家
            }
        }
    }
}
