# CAFD 工作备份

按内部工作备份保存原始源码、实验方法、结果和过程记录，未重新整理对外展示。

完整工作记录位于 backup/2026-09-08/ 的分块归档。恢复到新目录：

    python restore_backup.py --destination /path/to/new/CAFD

脚本核对所有分块及完整归档 SHA256 后按原路径恢复。逐文件清单为 source-files.jsonl 和 source-SHA256SUMS，原 Git 历史为 original-history.bundle。
模型权重、优化器恢复文件、二进制 probe 缓存、运行环境和符号链接本体未入包，保留索引见 excluded-files.jsonl；第三方源码版本见 dependency_sources.json。
现有实验报告及其归档原样保留。tmp/ 和验证目录中的合成测试夹具也按原路径备份，不能当作正式实验输出。
