/*
 * Standalone shim for tensorrt_llm/common/tllmException.h
 *
 * Minimal TllmException + TLLM_THROW macro suitable for the standalone
 * blockscale_gemm benchmark build. Stack-trace capture is omitted; the
 * exception is otherwise drop-in for code that only `throw`s and `what()`s.
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <typeinfo>

#define NEW_TLLM_EXCEPTION(...)                                                                                        \
    tensorrt_llm::common::TllmException(__FILE__, __LINE__, tensorrt_llm::common::fmtstr(__VA_ARGS__).c_str())

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define TLLM_WRAP(ex)                                                                                                  \
    NEW_TLLM_EXCEPTION("%s: %s", typeid(ex).name(), ex.what())

TRTLLM_NAMESPACE_BEGIN
namespace common
{

class TllmException : public std::runtime_error
{
public:
    TllmException(char const* file, std::size_t line, char const* msg)
        : std::runtime_error(buildMessage(file, line, msg))
    {
    }

    static std::string demangle(char const* name)
    {
        return std::string(name ? name : "");
    }

private:
    static std::string buildMessage(char const* file, std::size_t line, char const* msg)
    {
        std::string out = "[blockscale_gemm][ERROR] ";
        if (file != nullptr)
        {
            out += file;
            out += ":";
            out += std::to_string(line);
            out += ": ";
        }
        out += (msg != nullptr ? msg : "<no message>");
        return out;
    }
};

[[noreturn]] inline void throwRuntimeError(char const* file, int line, char const* info)
{
    throw TllmException(file, static_cast<std::size_t>(line),
        fmtstr("Assertion failed: %s", info != nullptr ? info : "").c_str());
}

[[noreturn]] inline void throwRuntimeError(char const* file, int line, std::string const& info = "")
{
    throwRuntimeError(file, line, info.c_str());
}

} // namespace common
TRTLLM_NAMESPACE_END
