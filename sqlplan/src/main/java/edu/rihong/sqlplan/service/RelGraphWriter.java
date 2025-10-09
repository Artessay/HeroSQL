
package edu.rihong.sqlplan.service;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.util.Pair;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.io.PrintWriter;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Implementation of {@link org.apache.calcite.rel.RelWriter}.
 */
public class RelGraphWriter extends RelWriterImpl {

  //~ Constructors -----------------------------------------------------------

  public RelGraphWriter(PrintWriter pw) {
    super(pw, SqlExplainLevel.EXPPLAN_ATTRIBUTES, false);
  }

  //~ Methods ----------------------------------------------------------------

  @Override protected void explain_(RelNode rel,
      List<Pair<String, @Nullable Object>> values) {
    List<RelNode> inputs = rel.getInputs();
    final RelMetadataQuery mq = rel.getCluster().getMetadataQuery();
    if (!mq.isVisibleInExplain(rel, detailLevel)) {
      // render children in place of this, at same level
      explainInputs(inputs);
      return;
    }

    StringBuilder s = new StringBuilder();
    spacer.spaces(s);
    if (withIdPrefix) {
      s.append(rel.getId()).append(":");
    }
    s.append(rel.getRelTypeName());

    if (detailLevel != SqlExplainLevel.NO_ATTRIBUTES) {
      values = replaceRexInputRef(rel, values);
 
      int j = 0;
      for (Pair<String, @Nullable Object> value : values) {
        if (value.right instanceof RelNode) {
          continue;
        }
        if (j++ == 0) {
          s.append("(");
        } else {
          s.append(", ");
        }
        s.append(value.left)
            .append("=[")
            .append(value.right)
            .append("]");
      }
      if (j > 0) {
        s.append(")");
      }
    }
    switch (detailLevel) {
    case ALL_ATTRIBUTES:
      s.append(": rowcount = ")
          .append(mq.getRowCount(rel))
          .append(", cumulative cost = ")
          .append(mq.getCumulativeCost(rel));
      break;
    default:
      break;
    }
    switch (detailLevel) {
    case NON_COST_ATTRIBUTES:
    case ALL_ATTRIBUTES:
      if (!withIdPrefix) {
        // If we didn't print the rel id at the start of the line, print
        // it at the end.
        s.append(", id = ").append(rel.getId());
      }
      break;
    default:
      break;
    }
    pw.println(s);
    spacer.add(2);
    explainInputs(inputs);
    spacer.subtract(2);
  }

  private void explainInputs(List<RelNode> inputs) {
    for (RelNode input : inputs) {
      input.explain(this);
    }
  }

  protected static List<Pair<String, @Nullable Object>> replaceRexInputRef(
      RelNode rel, 
      List<Pair<String, @Nullable Object>> valueList) {
      List<String> inputFieldNames = getInputFieldNames(rel);

      return valueList.stream()
          .map(value -> {
            if (value.right instanceof RexInputRef) {
              RexInputRef inputRef = (RexInputRef) value.right;
              String inputFieldName = inputFieldNames.get(inputRef.getIndex());
              
              // if inputFieldName contains space, surround it with quotes
              inputFieldName = inputFieldName.contains(" ") ? "`" + inputFieldName + "`" : inputFieldName;

              return Pair.<String, @Nullable Object>of(value.left, inputFieldName);
            } else if (!(value.right instanceof RelNode) ) {
              String str = value.right.toString();
              str = replacePlaceholders(str, inputFieldNames);
              return Pair.<String, @Nullable Object>of(value.left, str);
            } else {
              return value;
            }
          })
          .collect(Collectors.toList());
  }

  private static String replacePlaceholders(String str, List<String> inputFieldNames) {
    if (str == null) {
      return null;
    }
    if (inputFieldNames == null || inputFieldNames.isEmpty()) {
      return str;
    }

    Pattern pattern = Pattern.compile("\\$(\\d+)");
    Matcher matcher = pattern.matcher(str);
    StringBuffer sb = new StringBuffer();
    while (matcher.find()) {
        int index = Integer.parseInt(matcher.group(1));
        String replacement = index >= 0 && index < inputFieldNames.size() ? inputFieldNames.get(index) : matcher.group(0);

        // if replacement contains space, surround it with quotes
        replacement = replacement.contains(" ") ? "`" + replacement + "`" : replacement;

        matcher.appendReplacement(sb, Matcher.quoteReplacement(replacement));
    }
    matcher.appendTail(sb);
    return sb.toString();
  }

  /**
   * Retrieves the field names from the inputs of the given {@link RelNode}.
   * <p>
   * This method collects all field names from all input nodes' row types into a single list.
   *
   * @param rel the relational node whose input field names to extract
   * @return a list of all field names from all of the rel's input nodes
   */
  private static List<String> getInputFieldNames(RelNode rel) {
    // Flatten field names from all input nodes' row types
    return rel.getInputs().stream()
        .map(RelNode::getRowType)
        .flatMap(rowType -> rowType.getFieldNames().stream())
        .collect(Collectors.toList());
  }

}
