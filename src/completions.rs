pub fn script(shell: &str) -> Option<&'static str> {
    match shell {
        "bash" => Some(BASH),
        "zsh" => Some(ZSH),
        "fish" => Some(FISH),
        _ => None,
    }
}

const BASH: &str = r#"# cubi bash completions
_cubi() {
    local cur
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"

    if [[ "${COMP_WORDS[1]}" == "completions" && ${COMP_CWORD} -eq 2 ]]; then
        COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") )
        return 0
    fi
    if [[ "${COMP_WORDS[1]}" == "plugins" && ${COMP_CWORD} -eq 2 ]]; then
        COMPREPLY=( $(compgen -W "list reload" -- "$cur") )
        return 0
    fi

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "--version -V -v version --help -h help --resume -r resume --stream --no-stream --markdown --no-markdown --show-stats-footer --system --json completions plugins" -- "$cur") )
    fi
}
complete -F _cubi cubi
"#;

const ZSH: &str = r#"#compdef cubi
# cubi zsh completions
_cubi() {
    local -a top shells
    top=(
        '--version:print version and exit'
        '-V:print version and exit'
        '-v:print version and exit'
        'version:print version and exit'
        '--help:print help and exit'
        '-h:print help and exit'
        'help:print help and exit'
        '--resume:resume a prior chat'
        '-r:resume a prior chat'
        'resume:resume a prior chat'
        '--stream:stream tokens live'
        '--no-stream:wait for the full reply'
        '--markdown:enable markdown rendering'
        '--no-markdown:disable markdown rendering'
        '--show-stats-footer:print token and timing stats'
        'completions:print a shell completion script'
        'plugins:manage plugin bundles'
    )
    plugins=(
        'list:list discovered plugin bundles'
        'reload:rediscover plugin bundles'
    )
    shells=(
        'bash:Bash completion script'
        'zsh:Zsh completion script'
        'fish:Fish completion script'
    )

    if (( CURRENT == 2 )); then
        _describe -t cubi-commands 'cubi command' top
    elif [[ ${words[2]} == completions && CURRENT -eq 3 ]]; then
        _describe -t shells 'shell' shells
    elif [[ ${words[2]} == plugins && CURRENT -eq 3 ]]; then
        _describe -t plugins 'plugin subcommand' plugins
    fi
}
_cubi "$@"
"#;

const FISH: &str = r#"# cubi fish completions
complete -c cubi -f
complete -c cubi -l version -s V -d 'Print version and exit'
complete -c cubi -s v -d 'Print version and exit'
complete -c cubi -l help -s h -d 'Print help and exit'
complete -c cubi -l resume -s r -d 'Resume a prior chat'
complete -c cubi -n '__fish_use_subcommand' -a version -d 'Print version and exit'
complete -c cubi -n '__fish_use_subcommand' -a help -d 'Print help and exit'
complete -c cubi -n '__fish_use_subcommand' -a resume -d 'Resume a prior chat'
complete -c cubi -l stream -d 'Stream tokens live'
complete -c cubi -l no-stream -d 'Wait for the full reply'
complete -c cubi -l markdown -d 'Enable markdown rendering'
complete -c cubi -l no-markdown -d 'Disable markdown rendering'
complete -c cubi -l show-stats-footer -d 'Print token and timing stats'
complete -c cubi -l system -r -d 'Prepend a system prompt file'
complete -c cubi -l json -d 'Emit machine-readable output where supported'
complete -c cubi -n '__fish_use_subcommand' -a completions -d 'Print a shell completion script'
complete -c cubi -n '__fish_use_subcommand' -a plugins -d 'Manage plugin bundles'
complete -c cubi -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'
complete -c cubi -n '__fish_seen_subcommand_from plugins' -a 'list reload'
"#;
