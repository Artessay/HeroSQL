package edu.rihong.sqlplan.service;

import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.lookup.Lookup;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.springframework.stereotype.Service;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.Properties;

@Service
public class SqlPlanService {

    /**
     * 获取逻辑计划，返回为 plain text 形式
     */
    public String getLogicalPlanText(String dbPath, String sql) throws Exception {
        RelNode logicalPlan = getLogicalPlanNode(dbPath, sql);
        return toPlanString(logicalPlan);
    }

    /**
     * 获取逻辑计划，返回为 XML 形式
     */
    public String getLogicalPlanXml(String dbPath, String sql) throws Exception {
        RelNode logicalPlan = getLogicalPlanNode(dbPath, sql);
        return toPlanXmlString(logicalPlan);
    }

    /**
     * 获取逻辑计划，返回为 JSON 形式
     */
    public String getLogicalPlanJson(String dbPath, String sql) throws Exception {
        RelNode logicalPlan = getLogicalPlanNode(dbPath, sql);
        // 使用 RelOptUtil 的 JSON 格式（其实是带有属性的文本格式，可扩展为真正json）
        String jsonPlan = RelOptUtil.dumpPlan(
                "",
                logicalPlan,
                org.apache.calcite.sql.SqlExplainFormat.JSON,
                org.apache.calcite.sql.SqlExplainLevel.EXPPLAN_ATTRIBUTES
        );
        return jsonPlan;
    }

    // ----------------- 私有逻辑 -------------------------------

    private RelNode getLogicalPlanNode(String dbPath, String sql) throws Exception {
        String model = buildCalciteModel(dbPath);

        Properties info = new Properties();
        info.put("model", "inline:" + model);
        Connection conn = DriverManager.getConnection("jdbc:calcite:", info);
        try {
            CalciteConnection calciteConn = conn.unwrap(CalciteConnection.class);

            // 构建 Calcite parser 配置
            SqlParser.Config parserConfig = SqlParser.Config.DEFAULT
                    .withConformance(SqlConformanceEnum.BABEL)
                    .withCaseSensitive(false);

            // 构建 Calcite schema 配置
            SchemaPlus rootSchema = calciteConn.getRootSchema();
            Lookup<? extends SchemaPlus> lookup = rootSchema.subSchemas();
            SchemaPlus mainSchema = lookup.get("main");

            // 构建 Calcite 框架配置
            FrameworkConfig config = Frameworks.newConfigBuilder()
                    .parserConfig(parserConfig)
                    .defaultSchema(mainSchema)
                    .build();

            Planner planner = Frameworks.getPlanner(config);
            SqlNode parsed = planner.parse(sql);
            SqlNode validated = planner.validate(parsed);
            RelRoot relRoot = planner.rel(validated);
            return relRoot.project();
        } finally {
            conn.close();
        }
    }

    private String buildCalciteModel(String dbPath) {
        return "{\n" +
                "  \"version\": \"1.0\",\n" +
                "  \"defaultSchema\": \"main\",\n" +
                "  \"schemas\": [\n" +
                "    {\n" +
                "      \"name\": \"main\",\n" +
                "      \"type\": \"custom\",\n" +
                "      \"factory\": \"org.apache.calcite.adapter.jdbc.JdbcSchema$Factory\",\n" +
                "      \"operand\": {\n" +
                "        \"jdbcDriver\": \"org.sqlite.JDBC\",\n" +
                "        \"jdbcUrl\":    \"jdbc:sqlite:" + dbPath.replace("\\", "\\\\") + "\"\n" +
                "      }\n" +
                "    }\n" +
                "  ]\n" +
                "}";
    }

    /**
     * 以GraphWriter输出rel plan
     */
    private static String toPlanString(final RelNode rel) {
        if (rel == null) {
            return null;
        }
        final StringWriter sw = new StringWriter();
        final org.apache.calcite.rel.RelWriter planWriter =
                new RelGraphWriter(new PrintWriter(sw));
        rel.explain(planWriter);
        return sw.toString();
    }

    /**
     * 以GraphXmlWriter输出rel plan
     */
    private static String toPlanXmlString(final RelNode rel) {
        if (rel == null) {
            return null;
        }
        final StringWriter sw = new StringWriter();
        final org.apache.calcite.rel.RelWriter planWriter =
                new RelGraphXmlWriter(new PrintWriter(sw));
        rel.explain(planWriter);
        return sw.toString();
    }
}