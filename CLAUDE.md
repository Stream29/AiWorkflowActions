## 类型检查

```bash
ty check --exclude resources
```

保证每次编辑后都能通过类型检查，运行上面的命令。

## 依赖管理

```bash
uv run <script>
```

```bash
uv sync
```

```bash
uv add <package>
```

尽量使用`uv`，不要使用`pip`或者`python`。